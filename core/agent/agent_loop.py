"""
AgentLoopEngine — executes the mandatory cycle:
  analyze → plan → evaluate_risk → [confirm] → execute → observe → reflect → decide

Contract:
  - Max 5 iterations per goal
  - Every step is logged
  - Hard stop conditions enforced
  - Interrupt flag checked between steps
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import httpx

from core.state_machine import StateMachine, AgentState
from core.task_planner import TaskPlanner, Plan, PlanStep
from core.tool_router import ToolRouter, ToolObservation
from core.risk_evaluator import RiskEvaluator, RiskLevel
from core.autonomy_governor import AutonomyGovernor

logger = logging.getLogger("Jarvis.AgentLoop")

MAX_ITERATIONS = 5

REFLECT_SYSTEM_PROMPT = """You are Jarvis, a helpful AI assistant. 
Given the user's goal, the plan executed, and observations collected, 
produce a brief reflection and final response to the user.
Be concise, factual, and helpful. Speak directly to the user."""


@dataclass
class ExecutionTrace:
    goal: str
    iterations: int = 0
    plan: Optional[dict] = None
    observations: list[dict] = field(default_factory=list)
    risk_scores: list[dict] = field(default_factory=list)
    reflection: Optional[str] = None
    final_response: str = ""
    success: bool = False
    stop_reason: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def close(self, success: bool, reason: str):
        self.success = success
        self.stop_reason = reason
        self.ended_at = time.time()

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "iterations": self.iterations,
            "plan": self.plan,
            "observations": self.observations,
            "risk_scores": self.risk_scores,
            "reflection": self.reflection,
            "final_response": self.final_response,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "duration_seconds": round((self.ended_at or time.time()) - self.started_at, 2),
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
    ):
        self.sm = state_machine
        self.planner = task_planner
        self.router = tool_router
        self.risk = risk_evaluator
        self.gov = autonomy_governor
        self.model = model
        self.ollama_url = ollama_url
        self._interrupt = asyncio.Event()

    def request_interrupt(self):
        """Signal the loop to stop at next checkpoint."""
        self._interrupt.set()
        logger.info("Interrupt requested.")

    def _check_interrupt(self) -> bool:
        return self._interrupt.is_set()

    async def run(
        self,
        goal: str,
        context: str = "",
        confirm_callback=None,
    ) -> ExecutionTrace:
        """
        Execute the full agent loop for a given goal.
        confirm_callback(prompt: str) -> bool: called when user confirmation is needed.
        """
        self._interrupt.clear()
        trace = ExecutionTrace(goal=goal)
        self.router.reset_call_count()

        logger.info(f"=== Agent Loop START: {goal!r} ===")

        # ── PHASE 1: THINKING / PLANNING ──────────────────────────────────────
        self.sm.transition(AgentState.THINKING)
        if self._check_interrupt():
            return self._stop(trace, "user_interrupt")

        self.sm.transition(AgentState.PLANNING)
        plan = await self.planner.plan(goal, context)
        if not plan:
            self.sm.transition(AgentState.ERROR)
            self.sm.force_idle()
            trace.close(False, "planning_failed")
            trace.final_response = "I couldn't generate a plan for that goal. Please try rephrasing."
            return trace

        trace.plan = plan.raw_json
        logger.info(f"\n{plan.summary()}")

        if self._check_interrupt():
            return self._stop(trace, "user_interrupt")

        # ── PHASE 2: STEP EXECUTION LOOP ──────────────────────────────────────
        completed_steps: list[PlanStep] = []
        observations: list[ToolObservation] = []

        for iteration, step in enumerate(plan.steps):
            if trace.iterations >= MAX_ITERATIONS:
                return self._stop(trace, "iteration_limit_reached")

            if self._check_interrupt():
                return self._stop(trace, "user_interrupt")

            trace.iterations += 1
            logger.info(f"--- Step {step.step_id}/{len(plan)}: {step.description}")

            # Risk evaluation
            self.sm.transition(AgentState.RISK_EVALUATION)
            risk_report = self.risk.evaluate(
                goal=goal,
                tool_name=step.tool_required,
                step_description=step.description,
            )
            trace.risk_scores.append({
                "step_id": step.step_id,
                "score": risk_report.composite_score,
                "level": risk_report.level.value,
            })

            if risk_report.level == RiskLevel.BLOCK:
                logger.warning(f"Step {step.step_id} BLOCKED: {risk_report.explanation}")
                return self._stop(trace, "risk_threshold_exceeded")

            # No tool → reasoning step only
            if not step.tool_required:
                logger.info(f"Reasoning step (no tool): {step.description}")
                completed_steps.append(step)
                continue

            # Autonomy check
            allowed, reason = self.gov.can_execute(step.tool_required)
            if not allowed:
                logger.info(f"Tool not executed: {reason}")
                obs_text = f"[SUGGESTED] {step.description} → {reason}"
                trace.observations.append({"step_id": step.step_id, "note": obs_text})
                completed_steps.append(step)
                continue

            # Confirmation if required
            if risk_report.level in (RiskLevel.REVIEW, RiskLevel.CONFIRM) or \
               self.gov.requires_confirmation(step.tool_required):
                self.sm.transition(AgentState.AWAITING_CONFIRMATION)
                prompt = (
                    f"\nStep {step.step_id}: {step.description}\n"
                    f"Tool: {step.tool_required} | Risk: {risk_report.composite_score:.2f} [{risk_report.level.value}]\n"
                    f"Allow this action? [y/N]: "
                )
                approved = True  # default
                if confirm_callback:
                    approved = await confirm_callback(prompt)
                else:
                    # Fallback: blocking input in async context
                    answer = await asyncio.get_event_loop().run_in_executor(None, input, prompt)
                    approved = answer.strip().lower() in ("y", "yes")

                if not approved:
                    logger.info(f"Step {step.step_id} rejected by user.")
                    return self._stop(trace, "user_interrupt")

            # Execute
            self.sm.transition(AgentState.ACTING)
            args = step.arguments or {}
            obs = await self.router.execute(step.tool_required, args)

            # Observe
            self.sm.transition(AgentState.OBSERVING)
            trace.observations.append(obs.to_dict())
            observations.append(obs)

            if obs.execution_status == "failure":
                logger.error(f"Step {step.step_id} failed: {obs.error_message}")
                return self._stop(trace, "unrecoverable_tool_failure")

            completed_steps.append(step)

            if self._check_interrupt():
                return self._stop(trace, "user_interrupt")

        # ── PHASE 3: REFLECT & RESPOND ────────────────────────────────────────
        self.sm.transition(AgentState.REFLECTING)
        response = await self._reflect(goal, plan, observations)
        trace.reflection = response
        trace.final_response = response
        trace.close(True, "goal_completed")

        self.sm.transition(AgentState.SPEAKING)
        self.sm.transition(AgentState.IDLE)
        logger.info("=== Agent Loop COMPLETE ===")
        return trace

    def _stop(self, trace: ExecutionTrace, reason: str) -> ExecutionTrace:
        logger.info(f"Agent loop stopped: {reason}")
        trace.close(False, reason)
        if not trace.final_response:
            msgs = {
                "user_interrupt": "Understood — I've stopped.",
                "risk_threshold_exceeded": "I can't proceed — the risk level is too high for that action.",
                "iteration_limit_reached": "I've reached my step limit. Here's what I completed so far.",
                "unrecoverable_tool_failure": "A tool failed and I can't continue safely.",
                "planning_failed": "I couldn't create a plan for that.",
            }
            trace.final_response = msgs.get(reason, "Stopped.")
        self.sm.force_idle()
        return trace

    async def _reflect(self, goal: str, plan: Plan, observations: list[ToolObservation]) -> str:
        obs_text = "\n".join(
            f"- {o.tool_name}: {o.output_summary}" for o in observations
        ) or "No tools were executed (reasoning/suggest-only mode)."

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Plan summary:\n{plan.summary()}\n\n"
            f"Tool observations:\n{obs_text}\n\n"
            f"Please provide a helpful response to the user about what was accomplished."
        )

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
                resp = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Reflection LLM call failed: {e}")
            return "I completed the task. Check the execution trace for details."

