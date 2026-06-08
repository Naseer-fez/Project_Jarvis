"""Agent loop engine: plan -> risk -> confirm -> execute -> reflect."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore[assignment]

from core.state_machine import State as AgentState, StateMachine
from core.context.context import TaskExecutionContext
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.risk_evaluator import RiskEvaluator
from core.planner.planner import TaskPlanner
from core.metrics.confidence import ConfidenceModel
from core.registry.registry import ToolObservation, CapabilityRegistry

logger = logging.getLogger("Jarvis.AgentLoop")

_DEFAULT_MAX_ITERATIONS = 10

REFLECT_SYSTEM_PROMPT = (
    "You are Jarvis, an expert AI assistant. Review the executed plan and observations.\n"
    "If any tool failed: state the root cause first, then the fix.\n"
    "If successful: summarize concisely what was accomplished.\n"
    "Be direct and technical. No filler phrases. Address the user in second person."
)


def _truncate_obs(text: str, max_chars: int = 800) -> str:
    """Truncate long observations to keep both leading and trailing context."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return text[:half] + f"\n...[{omitted} chars omitted]...\n" + text[-half:]


def _truncate_observation(text: str, max_chars: int = 800) -> str:
    return _truncate_obs(text, max_chars=max_chars)


@dataclass
class ExecutionTrace:
    goal: str
    iterations: int = 0
    plan: Optional[dict[str, Any]] = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    risk_scores: list[dict[str, Any]] = field(default_factory=list)
    think_blocks: list[str] = field(default_factory=list)
    reflection: Optional[str] = None
    final_response: str = ""
    success: bool = False
    stop_reason: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def close(self, success: bool, reason: str) -> None:
        self.success = success
        self.stop_reason = reason
        self.ended_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "iterations": self.iterations,
            "plan": self.plan,
            "observations": self.observations,
            "risk_scores": self.risk_scores,
            "think_blocks": self.think_blocks,
            "reflection": self.reflection,
            "final_response": self.final_response,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "duration_seconds": round((self.ended_at or time.time()) - self.started_at, 3),
        }


class AgentLoopEngine:
    def __init__(
        self,
        state_machine: StateMachine | None = None,
        task_planner: TaskPlanner | None = None,
        tool_router: CapabilityRegistry | None = None,
        risk_evaluator: RiskEvaluator | None = None,
        autonomy_governor: AutonomyGovernor | None = None,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        llm: Any = None,  # Optional[LLMClientV2] — avoids import cycle
        container: Any = None,
    ):
        self.container = container
        self.sm = state_machine or (container.resolve("state_machine") if container and container.has("state_machine") else None)
        self.planner = task_planner or (container.resolve("task_planner") if container and container.has("task_planner") else None)
        self.router = tool_router or (container.resolve("tool_router") if container and container.has("tool_router") else None)
        self.risk = risk_evaluator or (container.resolve("risk_evaluator") if container and container.has("risk_evaluator") else None)
        self.gov = autonomy_governor or (container.resolve("autonomy_governor") if container and container.has("autonomy_governor") else None)
        self.model = model or "deepseek-r1:8b"
        self.ollama_url = ollama_url
        self.max_iterations = max(1, int(max_iterations or _DEFAULT_MAX_ITERATIONS))
        self.llm = llm or (container.resolve("llm") if container and container.has("llm") else None)
        self.confidence = ConfidenceModel()
        self._interrupt = asyncio.Event()
        self._run_lock = asyncio.Lock()

    def request_interrupt(self) -> None:
        self._interrupt.set()

    def _check_interrupt(self) -> bool:
        return self._interrupt.is_set()

    async def run(
        self,
        goal: str,
        context: TaskExecutionContext,
        confirm_callback=None,
    ) -> ExecutionTrace:
        async with context:
            async with self._run_lock:
                if self._check_interrupt():
                    return self._stop(ExecutionTrace(goal=goal), "user_interrupt", context.state_machine)

                self._interrupt.clear()

                sm = context.state_machine

            trace = ExecutionTrace(goal=goal)
            logger.info("Agent loop start: %s", goal)

            try:
                self._ensure_thinking_state(sm)
                sm.transition(AgentState.PLANNING)

                context_str = context.get("context_block", "")
                plan = await self._build_plan(goal, context_str)
                if not plan:
                    trace.final_response = "I couldn't generate a plan for that goal."
                    return self._stop(trace, "planning_failed", sm)

                trace.plan = plan
                intent_score = getattr(plan, "confidence", None)
                if intent_score is None and isinstance(plan, dict):
                    intent_score = plan.get("confidence", plan.get("intent_confidence", 0.5))
                try:
                    intent_score_value = float(intent_score) if intent_score is not None else 0.5
                except (TypeError, ValueError):
                    intent_score_value = 0.5
                self.confidence.update("intent_clarity", intent_score_value)

                if plan.get("clarification_needed"):
                    trace.final_response = str(
                        plan.get("clarification_prompt")
                        or plan.get("summary")
                        or "I need clarification before I can continue."
                    )
                    return self._stop(trace, "clarification_needed", sm)

                sm.transition(AgentState.RISK_EVALUATION)

                if self.risk is None:
                    raise RuntimeError("risk_evaluator is required but not provided.")
                plan_risk = self.risk.evaluate_plan(plan)
                trace.risk_scores.append(
                    {
                        "scope": "plan",
                        "level": plan_risk.level.label(),
                        "blocking": list(plan_risk.blocking_actions),
                        "confirm": list(plan_risk.confirm_actions),
                        "high": list(plan_risk.high_risk_actions),
                    }
                )

                if plan_risk.is_blocked:
                    trace.final_response = (
                        "I cannot execute that safely because the plan contains blocked actions: "
                        + ", ".join(plan_risk.blocking_actions)
                    )
                    sm.transition(AgentState.CANCELLED)
                    return self._stop(trace, "risk_threshold_exceeded", sm)

                from core.autonomy.autonomy_governor import AutonomyLevel
                is_autonomous = self.gov is not None and self.gov.level >= AutonomyLevel.AUTONOMOUS

                if plan_risk.requires_confirmation and not is_autonomous:
                    sm.transition(AgentState.AWAITING_CONFIRMATION)
                    approved = await self._ask_confirmation(
                        "This request includes high-impact actions. Continue? [y/N]: ",
                        confirm_callback,
                        context,
                    )
                    if not approved:
                        sm.transition(AgentState.CANCELLED)
                        return self._stop(trace, "user_interrupt", sm)
                    sm.transition(AgentState.APPROVED)
                else:
                    sm.transition(AgentState.APPROVED)

                # ── DAG Execution Engine integration (Session 8 Target Architecture) ──
                logger.info("Starting DAG Executor for goal: %s", goal, extra={"metadata": {"goal": goal}})
                
                if self.container and self.container.has("dag_executor"):
                    executor = self.container.resolve("dag_executor", tool_router=self.router, risk_evaluator=self.risk, autonomy_governor=self.gov)
                else:
                    raise RuntimeError("dag_executor not found in container")
                
                sm.transition(AgentState.EXECUTING)

                try:
                    # Enforce 5 minute task-level timeout (Part 5)
                    async with asyncio.timeout(300):
                        res = await executor.execute(plan, context)
                except asyncio.TimeoutError:
                    logger.error("Task execution timed out.", extra={"metadata": {"timeout_s": 300}})
                    res = {"status": "failure", "error": "Task execution timed out after 300s."}
                except Exception as exc:
                    logger.error("Execution engine failure: %s", exc, extra={"metadata": {"error": str(exc)}})
                    res = {"status": "failure", "error": str(exc)}

                # Map execution results back into trace observations
                observations: list[ToolObservation] = []
                for sid, step_res in res.get("results", {}).items():
                    obs = ToolObservation(
                        tool_name=step_res.get("tool_name", ""),
                        arguments=step_res.get("arguments", {}),
                        execution_status=step_res.get("execution_status", "failure"),
                        output_summary=step_res.get("output_summary", ""),
                        error_message=step_res.get("error_message"),
                        duration_seconds=step_res.get("duration_seconds", 0.0),
                    )
                    observations.append(obs)

                    obs_dict = obs.to_dict()
                    obs_dict["step_id"] = sid
                    obs_dict["output_summary"] = _truncate_obs(str(obs_dict.get("output_summary", "")))
                    if obs_dict.get("error_message"):
                        obs_dict["error_message"] = _truncate_obs(str(obs_dict["error_message"]))
                    trace.observations.append(obs_dict)

                    tool_success = 1.0 if obs.execution_status == "success" else 0.0
                    self.confidence.update("tool_reliability", tool_success)

                if res.get("status") == "success":
                    sm.transition(AgentState.REFLECTING)
                    response = await self._reflect(goal, plan, observations, trace)
                    trace.reflection = response
                    trace.final_response = response
                    trace.close(True, "goal_completed")
                    sm.transition(AgentState.SPEAKING)
                    sm.transition(AgentState.COMPLETED)
                    sm.transition(AgentState.IDLE)
                else:
                    trace.final_response = res.get("error") or "Task execution failed."
                    return self._stop(trace, "unrecoverable_tool_failure", sm)

                logger.info("Agent loop complete: success=%s", trace.success)
                return trace

            except asyncio.CancelledError:
                logger.warning("Agent loop cancelled via asyncio.CancelledError")
                if sm.state not in {AgentState.ABORTED, AgentState.ERROR, AgentState.SHUTDOWN}:
                    if sm.can_transition(AgentState.ABORTED):
                        sm.transition(AgentState.ABORTED)
                    else:
                        sm.force_idle()
                raise
            except Exception as e:
                import traceback
                logger.error("Agent loop crashed: %s\n%s", e, traceback.format_exc())
                if sm.state not in {AgentState.ERROR, AgentState.ABORTED, AgentState.SHUTDOWN}:
                    if sm.can_transition(AgentState.ERROR):
                        sm.transition(AgentState.ERROR)
                    else:
                        sm.force_idle()
                        sm.transition(AgentState.ERROR)
                trace.final_response = f"Internal error during execution: {e}"
                return self._stop(trace, "internal_error", sm)
            finally:
                if sm.state not in {AgentState.IDLE, AgentState.SHUTDOWN, AgentState.ERROR, AgentState.ABORTED}:
                    try:
                        sm.force_idle()
                    except Exception:
                        pass

    def _ensure_thinking_state(self, sm: StateMachine) -> None:
        if sm.state == AgentState.THINKING:
            return
        if sm.state == AgentState.IDLE:
            sm.transition(AgentState.THINKING)
            return
        sm.force_idle()
        sm.transition(AgentState.THINKING)

    async def _build_plan(self, goal: str, context: str) -> dict[str, Any]:
        plan_fn = getattr(self.planner, "plan", None)
        if plan_fn is None:
            return {}

        try:
            if inspect.iscoroutinefunction(plan_fn):
                result = await plan_fn(goal, context)
            else:
                result = await asyncio.to_thread(plan_fn, goal, context)
        except Exception as exc:  # noqa: BLE001
            logger.error("Planner execution failed: %s", exc, exc_info=True)
            return {}

        if isinstance(result, dict):
            return result
        return {}

    def _normalize_steps(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        if not isinstance(steps, list):
            return []

        normalized: list[dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            params = step.get("params")
            if not params:
                params = step.get("parameters") or step.get("args") or {}
            if not isinstance(params, dict):
                params = {}
            normalized.append(
                {
                    "id": int(step.get("id", idx)),
                    "action": str(step.get("action", "")).strip(),
                    "description": str(step.get("description", "")).strip(),
                    "params": params,
                }
            )
        return normalized

    async def _ask_confirmation(self, prompt: str, confirm_callback, context: TaskExecutionContext) -> bool:
        if context.get("approval_called"):
            logger.warning("Approval already handled. Returning idempotent result: %s", context.get("approval_result"))
            return bool(context.get("approval_result"))

        approved = False
        if confirm_callback is None:
            import sys
            from core.autonomy.autonomy_governor import AutonomyLevel
            
            is_headless = not sys.stdin.isatty()
            is_autonomous = self.gov is not None and self.gov.level >= AutonomyLevel.AUTONOMOUS
            
            if is_headless or is_autonomous:
                logger.warning("Bypassing manual confirmation due to headless environment or LEVEL_4 autonomy.")
                approved = True
            else:
                answer = await asyncio.to_thread(input, prompt)
                approved = str(answer).strip().lower() in {"y", "yes"}
        else:
            try:
                result = confirm_callback(prompt)
                if inspect.isawaitable(result):
                    result = await result
                approved = bool(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Confirmation callback failed: %s", exc)
                approved = False

        context.set("approval_called", True)
        context.set("approval_result", approved)
        return approved

    async def _reflect(
        self,
        goal: str,
        plan: dict[str, Any],
        observations: list[ToolObservation],
        trace: ExecutionTrace,
    ) -> str:
        if observations:
            obs_lines = []
            for obs in observations:
                obs_text = _truncate_obs(obs.output_summary or obs.error_message or "")
                obs_lines.append(f"- {obs.tool_name}: {obs_text}")
            obs_text = "\n".join(obs_lines)
        else:
            obs_text = "No tool observations."

        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Plan:\n{self._plan_summary(plan)}\n\n"
            f"Tool observations:\n{obs_text}\n"
        )

        # ── Prefer LLMClientV2 if injected ────────────────────────────────────
        if self.llm is not None and hasattr(self.llm, "complete"):
            try:
                result = await self.llm.complete(
                    user_prompt,
                    system=REFLECT_SYSTEM_PROMPT,
                    temperature=0.2,
                    task_type="reflection",
                )
                cleaned = re.sub(r"<think>.*?</think>", "", result or "", flags=re.DOTALL).strip()
                if cleaned:
                    return cleaned
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLMClientV2 reflection failed: %s", exc)
            # Fall through to httpx if LLMClientV2 returned empty

        # ── Fallback: direct httpx Ollama call ────────────────────────────────
        if httpx is None:
            return self._fallback_reflection(plan, observations)

        payload = {
            "model": self.model, # self.model now just acts as fallback if llm is missing
            "messages": [
                {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                raw = str(data.get("message", {}).get("content", "")).strip()

                think_matches = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
                trace.think_blocks = [block.strip() for block in think_matches if block.strip()]

                cleaned_response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return cleaned_response or self._fallback_reflection(plan, observations)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflection model call failed: %s", exc)
            return self._fallback_reflection(plan, observations)

    def _plan_summary(self, plan: dict[str, Any]) -> str:
        summary = str(plan.get("summary", "")).strip()
        steps = self._normalize_steps(plan)
        lines = [summary] if summary else []
        for step in steps:
            action = step["action"] or "(no action)"
            desc = step["description"] or ""
            lines.append(f"{step['id']}. {action} - {desc}")
        return "\n".join(lines) if lines else "No plan summary available."

    def _fallback_reflection(self, plan: dict[str, Any], observations: list[ToolObservation]) -> str:
        if any(obs.execution_status != "success" for obs in observations):
            failed = [obs.tool_name for obs in observations if obs.execution_status != "success"]
            return (
                "You hit a tool failure. Root cause: one or more tool calls returned an error. "
                f"Failed tools: {', '.join(failed)}. "
                "Fix the failing tool inputs or environment and run the task again."
            )

        if observations:
            tool_list = ", ".join(obs.tool_name for obs in observations)
            return f"You completed the requested task successfully. Tools used: {tool_list}."

        summary = str(plan.get("summary", "")).strip()
        return summary or "You completed the task successfully."

    def _stop(self, trace: ExecutionTrace, reason: str, sm: StateMachine) -> ExecutionTrace:
        trace.close(False, reason)
        if not trace.final_response:
            defaults = {
                "user_interrupt": "Understood. I stopped the task.",
                "planning_failed": "I couldn't create a workable plan.",
                "clarification_needed": "I need clarification before proceeding.",
                "risk_threshold_exceeded": "I can't continue because the requested action is blocked by safety rules.",
                "iteration_limit_reached": "I reached the maximum number of iterations for this task.",
                "unrecoverable_tool_failure": "A required tool failed, so I stopped execution.",
            }
            trace.final_response = defaults.get(reason, "Task stopped.")

        sm.force_idle()
        return trace


__all__ = ["AgentLoopEngine", "ExecutionTrace", "_truncate_observation"]
