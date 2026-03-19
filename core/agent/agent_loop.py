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

from core.agent.state_machine import AgentState, StateMachine
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.risk_evaluator import RiskEvaluator
from core.llm.task_planner import TaskPlanner
from core.metrics.confidence import ConfidenceModel
from core.tools.tool_router import ToolObservation, ToolRouter

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
        state_machine: StateMachine,
        task_planner: TaskPlanner,
        tool_router: ToolRouter,
        risk_evaluator: RiskEvaluator,
        autonomy_governor: AutonomyGovernor,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        llm: Any = None,  # Optional[LLMClientV2] — avoids import cycle
    ):
        self.sm = state_machine
        self.planner = task_planner
        self.router = tool_router
        self.risk = risk_evaluator
        self.gov = autonomy_governor
        self.model = model or "deepseek-r1:8b"
        self.ollama_url = ollama_url
        self.max_iterations = max(1, int(max_iterations or _DEFAULT_MAX_ITERATIONS))
        self.llm = llm  # LLMClientV2 instance, if provided
        self.confidence = ConfidenceModel()
        self._interrupt = asyncio.Event()

    def request_interrupt(self) -> None:
        self._interrupt.set()

    def _check_interrupt(self) -> bool:
        return self._interrupt.is_set()

    async def run(
        self,
        goal: str,
        context: str = "",
        confirm_callback=None,
    ) -> ExecutionTrace:
        if self._check_interrupt():
            return self._stop(ExecutionTrace(goal=goal), "user_interrupt")

        self._interrupt.clear()
        self.router.reset_call_count()

        trace = ExecutionTrace(goal=goal)
        logger.info("Agent loop start: %s", goal)

        self._ensure_thinking_state()
        self.sm.transition(AgentState.PLANNING)

        plan = await self._build_plan(goal, context)
        if not plan:
            trace.final_response = "I couldn't generate a plan for that goal."
            return self._stop(trace, "planning_failed")

        trace.plan = plan
        intent_score = getattr(plan, "confidence", None)
        if intent_score is None and isinstance(plan, dict):
            intent_score = plan.get("confidence", plan.get("intent_confidence", 0.5))
        try:
            intent_score_value = float(intent_score)
        except (TypeError, ValueError):
            intent_score_value = 0.5
        self.confidence.update("intent_clarity", intent_score_value)

        if plan.get("clarification_needed"):
            trace.final_response = str(
                plan.get("clarification_prompt")
                or plan.get("summary")
                or "I need clarification before I can continue."
            )
            return self._stop(trace, "clarification_needed")

        self.sm.transition(AgentState.RISK_EVALUATION)

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
            return self._stop(trace, "risk_threshold_exceeded")

        if plan_risk.requires_confirmation:
            approved = await self._ask_confirmation(
                "This request includes high-impact actions. Continue? [y/N]: ",
                confirm_callback,
            )
            if not approved:
                return self._stop(trace, "user_interrupt")

        steps = self._normalize_steps(plan)
        observations: list[ToolObservation] = []

        for step in steps:
            if self._check_interrupt():
                return self._stop(trace, "user_interrupt")

            if trace.iterations >= self.max_iterations:
                return self._stop(trace, "iteration_limit_reached")

            trace.iterations += 1
            step_id = step["id"]
            action = step["action"]
            description = step["description"]
            params = step["params"]

            step_risk = self.risk.evaluate([action] if action else [])
            trace.risk_scores.append(
                {
                    "scope": f"step:{step_id}",
                    "action": action,
                    "level": step_risk.level.label(),
                    "blocking": list(step_risk.blocking_actions),
                    "confirm": list(step_risk.confirm_actions),
                    "high": list(step_risk.high_risk_actions),
                }
            )

            if step_risk.is_blocked:
                trace.final_response = f"Stopped at step {step_id}: blocked action '{action}'."
                return self._stop(trace, "risk_threshold_exceeded")

            if not action:
                trace.observations.append(
                    {
                        "step_id": step_id,
                        "tool_name": "",
                        "execution_status": "success",
                        "output_summary": f"Reasoning step: {description}",
                    }
                )
                continue

            allowed, reason = self.gov.can_execute(action)
            if not allowed:
                trace.observations.append(
                    {
                        "step_id": step_id,
                        "tool_name": action,
                        "execution_status": "failure",
                        "output_summary": "",
                        "error_message": reason,
                    }
                )
                continue

            force_confirmation = self.confidence.score() < 0.4
            needs_confirm = (
                step_risk.requires_confirmation
                or self.gov.requires_confirmation(action)
                or force_confirmation
            )
            if needs_confirm:
                self.sm.transition(AgentState.AWAITING_CONFIRMATION)
                approved = await self._ask_confirmation(
                    f"Step {step_id}: run '{action}' with args {params}? [y/N]: ",
                    confirm_callback,
                )
                if not approved:
                    return self._stop(trace, "user_interrupt")

            if self.sm.state in {AgentState.RISK_EVALUATION, AgentState.AWAITING_CONFIRMATION, AgentState.OBSERVING}:
                self.sm.transition(AgentState.ACTING)

            observation = await self.router.execute(action, params)
            observations.append(observation)
            self.sm.transition(AgentState.OBSERVING)

            obs_dict = observation.to_dict()
            obs_dict["step_id"] = step_id
            obs_dict["output_summary"] = _truncate_obs(str(obs_dict.get("output_summary", "")))
            if obs_dict.get("error_message"):
                obs_dict["error_message"] = _truncate_obs(str(obs_dict["error_message"]))
            trace.observations.append(obs_dict)
            tool_success = 1.0 if observation.execution_status == "success" else 0.0
            self.confidence.update("tool_reliability", tool_success)

            if observation.execution_status != "success":
                return self._stop(trace, "unrecoverable_tool_failure")

        if self.sm.state == AgentState.RISK_EVALUATION:
            self.sm.transition(AgentState.ACTING)
            self.sm.transition(AgentState.OBSERVING)

        self.sm.transition(AgentState.REFLECTING)
        response = await self._reflect(goal, plan, observations, trace)
        trace.reflection = response
        trace.final_response = response
        trace.close(True, "goal_completed")

        self.sm.transition(AgentState.SPEAKING)
        self.sm.transition(AgentState.IDLE)

        logger.info("Agent loop complete: success=%s", trace.success)
        return trace

    def _ensure_thinking_state(self) -> None:
        if self.sm.state == AgentState.THINKING:
            return
        if self.sm.state == AgentState.IDLE:
            self.sm.transition(AgentState.THINKING)
            return
        self.sm.force_idle()
        self.sm.transition(AgentState.THINKING)

    async def _build_plan(self, goal: str, context: str) -> dict[str, Any]:
        plan_fn = getattr(self.planner, "plan", None)
        if plan_fn is None:
            return {}

        try:
            if inspect.iscoroutinefunction(plan_fn):
                result = await plan_fn(goal, context)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, plan_fn, goal, context)
        except Exception as exc:  # noqa: BLE001
            logger.error("Planner execution failed: %s", exc)
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
            params = step.get("params", {})
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

    async def _ask_confirmation(self, prompt: str, confirm_callback) -> bool:
        if confirm_callback is None:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, input, prompt)
            return str(answer).strip().lower() in {"y", "yes"}

        try:
            result = confirm_callback(prompt)
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Confirmation callback failed: %s", exc)
            return False

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
                    task_type="final_response",
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
            "model": self.model,
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

    def _stop(self, trace: ExecutionTrace, reason: str) -> ExecutionTrace:
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

        self.sm.force_idle()
        return trace


__all__ = ["AgentLoopEngine", "ExecutionTrace", "_truncate_observation"]
