"""
JARVIS Controller v2 - Session 5
Orchestrates: State Machine -> Task Planner -> Risk Evaluator -> Executor
Voice transcripts are automatically processed just like CLI input.
Trusted Core: No bypass of any component is possible.
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from core.state_machine import JarvisStateMachine, State
from core.risk_evaluator import RiskEvaluator, RiskResult
from tasks.task_planner import TaskPlanner

if TYPE_CHECKING:
    from memory.hybrid_memory import HybridMemory

logger = logging.getLogger("JARVIS.Controller")


class JarvisController:
    def __init__(self, state_machine: JarvisStateMachine, memory: "HybridMemory"):
        self.sm = state_machine
        self.memory = memory
        self.planner = TaskPlanner()
        self.risk = RiskEvaluator()
        self._initialized = False
        logger.info("Controller initialized.")

    async def initialize(self):
        # Pre-load any preferences
        prefs = self.memory.get_preferences()
        logger.info(f"Loaded {len(prefs)} user preferences from memory.")
        self._initialized = True

    async def process_input(self, user_input: str, source: str = "cli") -> str:
        """
        Full pipeline: IDLE -> PLANNING -> RISK_CHECK -> EXECUTING -> RESPONDING
        This method is called by both the CLI loop and the Voice Layer.
        """
        if not self.sm.is_idle():
            logger.warning(f"Received input while not IDLE (state={self.sm.state.name}). Queuing...")
            # Simple behavior: wait for idle
            for _ in range(20):
                await asyncio.sleep(0.5)
                if self.sm.is_idle():
                    break
            else:
                return "I'm currently busy. Please wait a moment and try again."

        try:
            # IDLE -> PLANNING
            if not self.sm.transition(State.PLANNING, f"Processing: '{user_input[:50]}'"):
                return "State error. Please try again."

            # Generate JSON plan via DeepSeek R1
            plan = await self.planner.plan(user_input)

            if plan.get("requires_clarification"):
                self.sm.transition(State.RESPONDING, "Clarification needed")
                response = plan.get("clarification_question", "Could you clarify your request?")
                self.sm.transition(State.IDLE, "Clarification sent")
                return response

            # PLANNING -> RISK_CHECK
            if not self.sm.transition(State.RISK_CHECK, f"Risk score: {plan}"):
                return "Risk check error."

            plan_text = str(plan)
            risk_result = self.risk.evaluate(plan_text, context=user_input)

            logger.info(f"Risk: score={risk_result.score} level={risk_result.level} approved={risk_result.approved}")

            task_id = self.memory.log_task(user_input, plan, risk_score=risk_result.score)

            if self.risk.is_blocked(risk_result):
                self.sm.transition(State.RESPONDING, "Blocked by risk evaluator")
                response = (
                    f"⚠️ I can't execute that plan. Risk level: {risk_result.level} "
                    f"(score: {risk_result.score}/100). "
                    f"Reason: {risk_result.reason}"
                )
                self.memory.update_task(task_id, "blocked", {"reason": risk_result.reason})
                self.sm.transition(State.IDLE, "Task blocked")
                return response

            if self.risk.require_confirmation(risk_result):
                # In voice mode we can't easily get confirmation — log and ask
                self.sm.transition(State.RESPONDING, "Confirmation required")
                response = (
                    f"This action has medium risk (score: {risk_result.score}/100): "
                    f"{risk_result.reason}. "
                    f"Say 'confirm' or type 'yes' to proceed, or 'cancel' to abort."
                )
                self.memory.update_task(task_id, "awaiting_confirmation")
                self.sm.transition(State.IDLE, "Awaiting confirmation")
                return response

            # RISK_CHECK -> EXECUTING
            if not self.sm.transition(State.EXECUTING, "Risk approved"):
                return "Execution state error."

            result = await self._execute_plan(plan)
            self.memory.update_task(task_id, "completed", result)

            # EXECUTING -> RESPONDING
            self.sm.transition(State.RESPONDING, "Execution complete")
            response = self._format_response(plan, result)

            self.sm.transition(State.IDLE, "Response sent")
            return response

        except Exception as e:
            logger.error(f"Controller pipeline error: {e}", exc_info=True)
            self.sm.force_idle()
            return f"I encountered an error: {str(e)}"

    async def _execute_plan(self, plan: dict) -> dict:
        """Execute the approved plan steps."""
        results = []
        steps = plan.get("steps", [])

        for step in steps:
            action = step.get("action", "unknown")
            params = step.get("parameters", {})
            logger.info(f"Executing step {step.get('step_id')}: {action}")

            step_result = await self._dispatch_action(action, params)
            results.append({"step": step.get("step_id"), "action": action, "result": step_result})

        return {"steps_executed": len(steps), "results": results}

    async def _dispatch_action(self, action: str, params: dict) -> dict:
        """Route actions to their handlers."""
        handlers = {
            "search_memory": self._action_search_memory,
            "get_system_info": self._action_system_info,
            "read_file": self._action_read_file,
            "list_directory": self._action_list_dir,
            "clarify_intent": self._action_clarify,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        else:
            logger.warning(f"No handler for action: {action}")
            return {"status": "skipped", "reason": f"Action '{action}' not implemented yet."}

    async def _action_search_memory(self, params: dict) -> dict:
        query = params.get("query", "")
        results = self.memory.semantic_search(query, n_results=3)
        return {"matches": len(results), "results": results}

    async def _action_system_info(self, params: dict) -> dict:
        import platform
        return {
            "os": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
        }

    async def _action_read_file(self, params: dict) -> dict:
        path = params.get("path", "")
        try:
            with open(path, 'r') as f:
                content = f.read(2000)  # Cap at 2KB
            return {"status": "ok", "content": content}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_list_dir(self, params: dict) -> dict:
        import os
        path = params.get("path", ".")
        try:
            items = os.listdir(path)
            return {"status": "ok", "items": items[:50]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_clarify(self, params: dict) -> dict:
        return {"status": "clarification", "message": params.get("question", "Could you be more specific?")}

    def _format_response(self, plan: dict, result: dict) -> str:
        intent = plan.get("intent", "your request")
        steps = result.get("steps_executed", 0)
        results = result.get("results", [])

        response_parts = [f"I've completed: {intent}."]

        for r in results:
            action = r.get("action")
            res = r.get("result", {})

            if action == "search_memory" and res.get("matches", 0) > 0:
                response_parts.append(f"Found {res['matches']} relevant memories.")
            elif action == "get_system_info":
                info = res
                response_parts.append(f"System: {info.get('os')} {info.get('machine')}.")
            elif action == "read_file":
                if res.get("status") == "ok":
                    response_parts.append("File read successfully.")
            elif action == "clarify_intent":
                return res.get("message", "Please clarify your request.")

        return " ".join(response_parts)

    async def run_cli_loop(self):
        """Text-based CLI loop — same pipeline as voice."""
        print("\n" + "="*50)
        print("JARVIS CLI Mode | Type 'quit' to exit")
        print("="*50)

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n👤 You: ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "bye"):
                    print("Goodbye.")
                    break

                response = await self.process_input(user_input, source="cli")
                print(f"\n🤖 JARVIS: {response}")

                self.memory.log_conversation(
                    user_input=user_input,
                    jarvis_reply=response,
                    source="cli"
                )

            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            except Exception as e:
                logger.error(f"CLI error: {e}")
