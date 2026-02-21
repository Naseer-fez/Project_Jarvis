"""
core/controller.py - Main orchestrator.
"""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Tuple

import core.logger as logger_mod
from core.execution.dispatcher import DispatchResult, ToolDispatcher
from core.hardware.serial_controller import SerialController
from core.memory.hybrid_memory import HybridMemory
from core.planning.task_planner import TaskPlanner
from core.risk_evaluator import RiskEvaluator
from core.state_machine import IllegalTransitionError, State, StateMachine
from core.tools.vision import VisionTool
from memory.short_term import ShortTermMemory

try:
    from colorama import Fore, Style
except ImportError:
    class _DummyColor:
        BLUE = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
        RESET_ALL = ""

    Fore = _DummyColor()
    Style = _DummyColor()

log = logging.getLogger("jarvis.controller")

ConfirmCallback = Callable[[dict, Any], Awaitable[bool]]


class Controller:
    def __init__(self, config: configparser.ConfigParser, voice: bool = False) -> None:
        self._config = config
        self._voice_on = voice

        self.fsm = StateMachine()
        self.memory = HybridMemory(config)
        self.short_memory = ShortTermMemory(max_turns=20)
        self.planner = TaskPlanner(config)
        self.risk = RiskEvaluator(config)
        self.vision = VisionTool(config)
        self.serial = SerialController(config=config)
        self.dispatcher = ToolDispatcher(
            config=config,
            memory=self.memory,
            vision=self.vision,
            serial_controller=self.serial,
        )

        self._voice_loop: Optional[Any] = None
        self.fsm.add_listener(self._on_state_change)
        self._running = False

    # -- Audit ------------------------------------------------------------

    def audit(self, event_type: str, payload: dict) -> str:
        return logger_mod.audit(event_type, payload)

    def _on_state_change(self, old: State, new: State) -> None:
        log.debug(f"State: {old.name} -> {new.name}")
        self.audit("STATE_TRANSITION", {"from": old.name, "to": new.name})

    # -- Planning ---------------------------------------------------------

    def plan_for_intent(self, intent: str) -> Tuple[dict, Any]:
        """
        Generate a plan for intent and evaluate its risk.
        Returns (plan_dict, RiskResult).
        """
        try:
            context_text = self.memory.recall(intent)
        except Exception as exc:
            log.warning(f"Memory recall failed during planning: {exc}")
            context_text = ""

        context = f"Known context:\n{context_text}" if context_text else ""
        plan = self.planner.plan(intent, context=context)

        actions = []
        for step in plan.get("steps", []):
            if isinstance(step, dict):
                action = step.get("action", "")
                if action:
                    actions.append(action)
        risk = self.risk.evaluate(actions)

        self.audit(
            "PLAN_GENERATED",
            {
                "intent": intent,
                "summary": plan.get("summary", ""),
                "steps": plan.get("steps", []),
                "risk_level": risk.level.name,
                "risk_blocked": risk.is_blocked,
                "risk_reasons": risk.reasons,
            },
        )
        return plan, risk

    async def execute_plan_with_feedback(
        self,
        intent: str,
        plan: dict,
        risk: Any,
        confirm_callback: Optional[ConfirmCallback] = None,
    ) -> tuple[str, list[DispatchResult]]:
        """
        Execute a planned workflow through dispatcher, then summarize outcomes.
        """
        if risk.is_blocked:
            response = f"I cannot execute this request. {risk.summary()}"
            self.audit(
                "PLAN_EXECUTION_BLOCKED",
                {"intent": intent, "risk_level": risk.level.name, "summary": risk.summary()},
            )
            return response, []

        if self._requires_confirmation(risk):
            approved = False
            if confirm_callback is not None:
                approved = await confirm_callback(plan, risk)
            else:
                approved = await self._confirm_plan(plan, risk)
            if not approved:
                self.audit(
                    "PLAN_EXECUTION_ABORTED",
                    {"intent": intent, "risk_level": risk.level.name},
                )
                return "Execution cancelled.", []

        if self.fsm.state == State.IDLE and self.fsm.can_transition(State.PLANNING):
            self.fsm.transition(State.PLANNING)

        try:
            self.fsm.transition(State.EXECUTING)
        except IllegalTransitionError:
            self.fsm.force_idle()
            raise

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.dispatcher.execute_plan, plan)

        self.audit(
            "PLAN_EXECUTED",
            {
                "intent": intent,
                "result_count": len(results),
                "results": [r.to_dict() for r in results],
            },
        )

        self._inject_execution_feedback(intent, results)
        summary = await loop.run_in_executor(None, self._summarize_execution, intent, plan, results)

        if self.fsm.state == State.EXECUTING and self.fsm.can_transition(State.SPEAKING):
            self.fsm.transition(State.SPEAKING)

        return summary, results

    def _requires_confirmation(self, risk: Any) -> bool:
        threshold_name = self._config.get(
            "risk", "voice_confirm_threshold", fallback="MEDIUM"
        ).upper()
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "FORBIDDEN": 3}
        required_at = order.get(threshold_name, 1)
        return order.get(risk.level.name, 0) >= required_at and not risk.is_blocked

    def _inject_execution_feedback(self, intent: str, results: list[DispatchResult]) -> None:
        ts = int(time.time())
        for i, res in enumerate(results, start=1):
            outcome = res.output if res.success else f"ERROR: {res.error}"
            short_line = f"tool[{res.action}] -> {outcome}"
            self.short_memory.add("assistant", short_line[:600])
            key = f"exec_{ts}_{i}_{res.action}"
            try:
                self.memory.store_fact(
                    key,
                    short_line[:2000],
                    source="tool_execution",
                    metadata={
                        "intent": intent,
                        "step_id": res.step_id,
                        "action": res.action,
                        "success": res.success,
                    },
                )
            except Exception as exc:
                log.warning(f"Failed to persist execution feedback: {exc}")

    def _summarize_execution(
        self,
        intent: str,
        plan: dict,
        results: list[DispatchResult],
    ) -> str:
        lines = []
        for res in results:
            if res.success:
                lines.append(f"- Step {res.step_id} [{res.action}] OK: {res.output[:300]}")
            else:
                lines.append(f"- Step {res.step_id} [{res.action}] FAILED: {res.error}")

        observation_block = "\n".join(lines) if lines else "- No tool steps executed."
        memory_context = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.short_memory.get_recent(8)
        )

        prompt = (
            f"User intent:\n{intent}\n\n"
            f"Plan summary:\n{plan.get('summary', '')}\n\n"
            f"Tool observations:\n{observation_block}\n\n"
            f"Recent short-term memory:\n{memory_context}\n\n"
            "Give a concise user-facing summary. Mention failures if any."
        )

        summary = self._call_ollama_summary(prompt)
        if summary:
            return summary
        return self._fallback_summary(results)

    def _call_ollama_summary(self, prompt: str) -> str:
        url = f"{self._config.get('ollama', 'base_url', fallback='http://localhost:11434')}/api/generate"
        payload = json.dumps(
            {
                "model": self._config.get("ollama", "planner_model", fallback="deepseek-r1:8b"),
                "prompt": prompt,
                "system": (
                    "You are Jarvis. Summarize execution results clearly and safely. "
                    "Do not output JSON. Keep it short."
                ),
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 200},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_s = int(self._config.get("ollama", "request_timeout_s", fallback="60"))
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            return str(data.get("response", "")).strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return ""
        except Exception:
            return ""

    def _fallback_summary(self, results: list[DispatchResult]) -> str:
        if not results:
            return "No actions were executed."
        ok = sum(1 for r in results if r.success)
        failed = len(results) - ok
        if failed == 0:
            return f"Execution complete. {ok} step(s) succeeded."
        return f"Execution finished with issues. {ok} succeeded, {failed} failed."

    # -- Startup / shutdown ----------------------------------------------

    async def start(self) -> None:
        self._running = True
        self.audit(
            "JARVIS_START",
            {
                "version": self._config.get("general", "version", fallback="2.0.0"),
                "voice": self._voice_on,
            },
        )

        try:
            if self.planner.ping():
                log.info("Ollama reachable")
            else:
                log.warning("Ollama not reachable - planner will degrade gracefully")
        except AttributeError:
            log.debug("Planner has no ping method, skipping Ollama check.")

        if self._voice_on:
            try:
                from core.voice.voice_loop import VoiceLoop

                self._voice_loop = VoiceLoop(self, self._config)
                self._voice_loop.start()
                log.info("Voice loop active")
            except Exception as exc:
                log.error(f"Voice loop failed to start: {exc} - falling back to CLI only")
                self._voice_on = False

        self._print_banner()

    async def shutdown(self) -> None:
        self._running = False
        log.info("Shutting down...")

        if self._voice_loop:
            await self._voice_loop.stop()

        self.audit("JARVIS_SHUTDOWN", {"ts": time.time()})
        try:
            self.fsm.force_idle()
        except Exception:
            pass
        log.info("Shutdown complete.")

    # -- CLI loop ---------------------------------------------------------

    async def run_cli(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                user_input = await loop.run_in_executor(None, self._prompt)
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            await self._handle_command(user_input)

    def _prompt(self) -> str:
        state = self.fsm.state.name
        return input(f"\n{Fore.BLUE}[{state}]{Style.RESET_ALL} You: ")

    async def _handle_command(self, text: str) -> None:
        lower = text.lower().strip()

        if lower in ("quit", "exit", "q"):
            self._running = False
            return
        if lower == "help":
            self._print_help()
            return
        if lower == "status":
            await self._cmd_status()
            return
        if lower == "memory":
            self._cmd_memory()
            return
        if lower == "history":
            self._cmd_history()
            return
        if lower == "reset":
            self._cmd_reset()
            return
        if lower.startswith("vision "):
            image_path = text[7:].strip()
            await self._cmd_vision(image_path)
            return

        await self._cmd_plan(text)

    # -- Commands ---------------------------------------------------------

    async def _cmd_plan(self, intent: str) -> None:
        try:
            self.fsm.transition(State.PLANNING)
        except IllegalTransitionError as exc:
            print(f"State error: Cannot plan right now ({exc}). Type 'reset' if stuck.")
            return

        print("Planning...")
        loop = asyncio.get_event_loop()
        try:
            plan, risk = await loop.run_in_executor(None, self.plan_for_intent, intent)
        except Exception as exc:
            log.error(f"Planning failed: {exc}", exc_info=True)
            try:
                self.fsm.transition(State.ERROR)
            except IllegalTransitionError:
                self.fsm.force_idle()
            print(f"Planning error: {exc}")
            return

        self._print_plan(plan, risk)
        try:
            summary, _ = await self.execute_plan_with_feedback(intent, plan, risk)
            print(f"\nJarvis: {summary}")
        except Exception as exc:
            log.error(f"Execution failed: {exc}", exc_info=True)
            print(f"\nExecution error: {exc}")
            if self.fsm.can_transition(State.ERROR):
                self.fsm.transition(State.ERROR)
        finally:
            self.fsm.force_idle()

    async def _confirm_plan(self, plan: dict, risk: Any) -> bool:
        del plan
        print(f"\n{Fore.YELLOW}Risk Level: {risk.level.name}{Style.RESET_ALL}")
        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: input("\nThis requires permission. Proceed? [y/N]: ").strip().lower(),
            )
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    async def _cmd_status(self) -> None:
        print(f"\n{'=' * 50}")
        print(f"  Jarvis v{self._config.get('general', 'version', fallback='2.0.0')}")
        print(f"{'=' * 50}")
        print(f"  FSM State:    {Fore.GREEN}{self.fsm.state.name}{Style.RESET_ALL}")
        try:
            print(f"  Ollama:       {'online' if self.planner.ping() else 'offline'}")
        except AttributeError:
            pass
        print(f"  Voice:        {'active' if self._voice_on else 'disabled'}")
        print(f"  Serial:       {'connected' if self.serial.is_connected else 'disconnected'}")
        ok, entries, err = logger_mod.verify_audit()
        audit_status = f"OK ({entries} entries)" if ok else f"TAMPERED - {err}"
        print(f"  Audit Log:    {audit_status}")
        print(f"{'=' * 50}\n")

    def _cmd_memory(self) -> None:
        print("To inspect memory details, query through a planning command or inspect data files.")

    def _cmd_history(self) -> None:
        audit_path = self._config.get("logging", "audit_file", fallback="logs/audit.jsonl")
        path = Path(audit_path)
        if not path.exists():
            print("No audit log yet.")
            return
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-20:]
        print(f"\n-- Last {len(recent)} audit entries --")
        for line in recent:
            try:
                entry = json.loads(line)
                payload = json.dumps(entry.get("payload", {}), ensure_ascii=False)[:120]
                print(f"  [{entry['ts'][11:19]}] {entry['event']}  {payload}")
            except Exception:
                print(f"  {line[:120]}")
        print()

    def _cmd_reset(self) -> None:
        try:
            self.fsm.force_idle()
            print("Reset to IDLE.")
            self.audit("MANUAL_RESET", {})
        except Exception as exc:
            print(f"Reset warning: {exc}")

    async def _cmd_vision(self, image_path: str) -> None:
        if not image_path:
            print("Usage: vision <path>")
            return
        print(f"Analyzing image: {image_path}")
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self.vision.analyze, image_path)
            print(f"\nVision result:\n{result}\n")
            self.audit("VISION_ANALYZE", {"path": image_path, "result_len": len(result)})
        except (FileNotFoundError, ValueError) as exc:
            print(f"Vision input error: {exc}")
        except Exception as exc:
            print(f"Vision error: {exc}")

    # -- Print helpers ----------------------------------------------------

    def _print_plan(self, plan: dict, risk: Any) -> None:
        print(f"\n{'-' * 50}")
        print(f"Plan: {plan.get('summary', '(no summary)')}")
        print(f"Confidence: {plan.get('confidence', 0.0):.0%} | Risk: {risk.level.name}")
        steps = plan.get("steps", [])
        if steps:
            print("Steps:")
            for s in steps:
                sid = s.get("id", "?")
                action = s.get("action", "?")
                desc = s.get("description", "")
                print(f"  {sid}. [{action}] {desc}")
        if plan.get("clarification_needed"):
            print(f"\nClarification: {plan.get('clarification_prompt', '')}")
        print(f"{'-' * 50}")

    def _print_banner(self) -> None:
        v = self._config.get("general", "version", fallback="2.0.0")
        mode = "VOICE + CLI" if self._voice_on else "CLI only"
        print(
            f"""
{Fore.BLUE}+--------------------------------------+
|          JARVIS  v{v:<16}|
|  Offline | Local | Human-in-loop      |
+--------------------------------------+{Style.RESET_ALL}
Mode: {mode}
Type 'help' for commands, 'quit' to exit.
"""
        )

    def _print_help(self) -> None:
        print(
            """
Commands:
  <any text>        Generate a plan and execute approved steps
  vision <path>     Analyze an image with LLaVA
  status            Show system status
  memory            Show memory hint
  history           Show recent audit log
  reset             Reset to IDLE
  help              Show this help
  quit              Shutdown
"""
        )
