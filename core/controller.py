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
from core.execution.async_task_manager import AsyncTaskManager
from core.execution.dispatcher import DispatchResult, ToolDispatcher
from core.hardware.serial_controller import SerialController
from core.memory.hybrid_memory import HybridMemory
from core.permission_matrix import PermissionMatrix
from core.planning.task_planner import TaskPlanner
from core.risk_evaluator import RiskEvaluator
from core.state_machine import IllegalTransitionError, State, StateMachine
from core.tools.registry import ToolRegistry
from core.tools.vision import VisionTool
from core.trace_logger import TraceLogger
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
        configured_session = config.get("general", "session_name", fallback="").strip()
        self.session_id = configured_session or f"session_{int(time.time())}"

        self.fsm = StateMachine()
        self.memory = HybridMemory(config)
        self.short_memory = ShortTermMemory(max_turns=20)
        self.planner = TaskPlanner(config)
        self.risk = RiskEvaluator(config)
        self.permissions = PermissionMatrix(config)
        self.vision = VisionTool(config)
        self.serial = SerialController(config=config)
        self.trace = TraceLogger(
            output_dir=config.get("logging", "trace_dir", fallback="outputs/Jarvis-Session"),
            session_id=self.session_id,
        )
        self.dispatcher = ToolDispatcher(
            config=config,
            memory=self.memory,
            vision=self.vision,
            serial_controller=self.serial,
            trace_logger=self.trace,
        )
        self.task_manager = AsyncTaskManager(
            max_parallel=int(config.get("concurrency", "max_parallel_tasks", fallback="3"))
        )
        self.tool_registry = ToolRegistry(
            enabled_scopes={
                s.strip()
                for s in config.get("plugins", "enabled_scopes", fallback="core").split(",")
                if s.strip()
            }
            or {"core"}
        )
        self._load_plugins()

        self._voice_loop: Optional[Any] = None
        self.fsm.add_listener(self._on_state_change)
        self._running = False
        self._current_dispatch_future: Optional[asyncio.Future] = None
        self._failsafe_error_count = 0
        self._failsafe_threshold = int(config.get("risk", "failsafe_error_threshold", fallback="3"))
        self._failsafe_auto_disable = config.getboolean(
            "risk", "failsafe_auto_disable_on_error", fallback=True
        )
        self._failsafe_disabled = False
        self._control_task: Optional[asyncio.Task] = None
        self._control_file = Path(
            config.get("dashboard", "control_file", fallback="runtime/control_flags.json")
        )
        self._control_file.parent.mkdir(parents=True, exist_ok=True)

    # -- Audit ------------------------------------------------------------

    def audit(self, event_type: str, payload: dict) -> str:
        return logger_mod.audit(event_type, payload)

    def _on_state_change(self, old: State, new: State) -> None:
        log.debug(f"State: {old.name} -> {new.name}")
        self.audit("STATE_TRANSITION", {"from": old.name, "to": new.name})
        self.trace.state(state=new.name.lower(), source="controller")

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

        actions: list[str] = []
        for step in plan.get("steps", []):
            if isinstance(step, dict):
                action = step.get("action", "")
                if action:
                    actions.append(str(action).strip().lower())
        risk = self.risk.evaluate(actions)
        perms = self.permissions.evaluate(actions)

        dedup_tools: list[str] = []
        seen: set[str] = set()
        for action in actions:
            if action not in seen:
                seen.add(action)
                dedup_tools.append(action)

        computed_risk = "critical" if perms.blocked_actions else risk.level.label().lower()
        confirmation_needed = bool(perms.confirmation_actions) or self._requires_confirmation(risk)
        plan["tools_required"] = dedup_tools
        plan["risk_level"] = computed_risk
        plan["confirmation_required"] = confirmation_needed

        self.audit(
            "PLAN_GENERATED",
            {
                "intent": intent,
                "summary": plan.get("summary", ""),
                "steps": plan.get("steps", []),
                "risk_level": computed_risk,
                "risk_blocked": risk.is_blocked,
                "risk_reasons": risk.reasons,
                "permission_blocked": perms.blocked_actions,
                "permission_confirm": perms.confirmation_actions,
            },
        )
        self.trace.decision(
            "plan_generated",
            intent=intent,
            summary=plan.get("summary", ""),
            tools_required=plan.get("tools_required", []),
            risk_level=plan.get("risk_level", "low"),
            confirmation_required=plan.get("confirmation_required", False),
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
        if self._failsafe_disabled:
            return "Failsafe mode is active. Action execution is temporarily disabled.", []

        actions = []
        for step in plan.get("steps", []):
            if isinstance(step, dict):
                action = str(step.get("action", "")).strip().lower()
                if action:
                    actions.append(action)
        perms = self.permissions.evaluate(actions)
        if perms.blocked_actions:
            response = (
                "I cannot execute this request. Blocked actions: "
                + ", ".join(perms.blocked_actions)
            )
            self.audit(
                "PLAN_EXECUTION_BLOCKED",
                {
                    "intent": intent,
                    "blocked_actions": perms.blocked_actions,
                    "reason": "permission_matrix",
                },
            )
            self.trace.decision(
                "plan_blocked",
                intent=intent,
                blocked_actions=perms.blocked_actions,
            )
            return response, []

        if risk.is_blocked:
            response = f"I cannot execute this request. {risk.summary()}"
            self.audit(
                "PLAN_EXECUTION_BLOCKED",
                {"intent": intent, "risk_level": risk.level.name, "summary": risk.summary()},
            )
            return response, []

        if self._requires_confirmation(risk) or perms.needs_confirmation:
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
        self.dispatcher.clear_cancel()
        dispatch_priority = int(plan.get("priority", 5))

        async def _dispatch_job():
            return await loop.run_in_executor(None, self.dispatcher.execute_plan, plan)

        task_id, dispatch_future = await self.task_manager.submit(
            _dispatch_job,
            priority=dispatch_priority,
            name="plan_dispatch",
        )
        self.trace.decision(
            "dispatch_submitted",
            task_id=task_id,
            priority=dispatch_priority,
        )
        self._current_dispatch_future = dispatch_future
        try:
            results = await self._current_dispatch_future
        except asyncio.CancelledError:
            self.dispatcher.request_cancel()
            self.trace.error("controller", "Dispatch future cancelled", intent=intent)
            return "Execution cancelled.", []
        except Exception as exc:
            self._failsafe_error_count += 1
            self.trace.error("controller", str(exc), intent=intent)
            if self._failsafe_auto_disable and self._failsafe_error_count >= self._failsafe_threshold:
                self._failsafe_disabled = True
                self.audit(
                    "FAILSAFE_ENABLED",
                    {
                        "reason": "execution_errors",
                        "error_count": self._failsafe_error_count,
                    },
                )
                self.trace.decision(
                    "failsafe_enabled",
                    reason="execution_errors",
                    error_count=self._failsafe_error_count,
                )
            raise
        finally:
            self._current_dispatch_future = None
            self.dispatcher.clear_cancel()

        self.audit(
            "PLAN_EXECUTED",
            {
                "intent": intent,
                "result_count": len(results),
                "results": [r.to_dict() for r in results],
            },
        )
        self.trace.decision(
            "plan_executed",
            intent=intent,
            result_count=len(results),
            failed=sum(1 for r in results if not r.success),
        )

        if any(not r.success for r in results):
            self._failsafe_error_count += 1
            if self._failsafe_auto_disable and self._failsafe_error_count >= self._failsafe_threshold:
                self._failsafe_disabled = True
                self.audit(
                    "FAILSAFE_ENABLED",
                    {
                        "reason": "step_failures",
                        "error_count": self._failsafe_error_count,
                    },
                )
                self.trace.decision(
                    "failsafe_enabled",
                    reason="step_failures",
                    error_count=self._failsafe_error_count,
                )

        self._inject_execution_feedback(intent, results)
        summary = await loop.run_in_executor(None, self._summarize_execution, intent, plan, results)

        if all(r.success for r in results):
            self._failsafe_error_count = 0

        if self.fsm.state == State.EXECUTING and self.fsm.can_transition(State.SPEAKING):
            self.fsm.transition(State.SPEAKING)

        return summary, results

    def _requires_confirmation(self, risk: Any) -> bool:
        threshold_name = self._config.get(
            "risk", "voice_confirm_threshold", fallback="MEDIUM"
        ).upper()
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "FORBIDDEN": 3}
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
                if hasattr(self.memory, "store_action"):
                    self.memory.store_action(
                        action=res.action,
                        outcome=outcome[:2000],
                        success=res.success,
                        metadata={"intent": intent, "step_id": res.step_id},
                    )
                if not res.success and hasattr(self.memory, "store_failure"):
                    self.memory.store_failure(
                        action=res.action,
                        error=str(res.error or ""),
                        metadata={"intent": intent, "step_id": res.step_id},
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

    def request_interrupt(self, reason: str = "manual") -> None:
        self.dispatcher.request_cancel()
        if self._current_dispatch_future is not None:
            self._current_dispatch_future.cancel()
        try:
            if self.fsm.state != State.IDLE and self.fsm.can_transition(State.INTERRUPTED):
                self.fsm.transition(State.INTERRUPTED)
                if self.fsm.can_transition(State.IDLE):
                    self.fsm.transition(State.IDLE)
        except Exception:
            self.fsm.force_idle()

        self.audit("INTERRUPT_REQUESTED", {"reason": reason})
        self.trace.decision("interrupt_requested", reason=reason)

    def _load_plugins(self) -> None:
        plugin_dir = self._config.get("plugins", "directory", fallback="core/plugins")
        loaded = self.tool_registry.load_plugins(plugin_dir)
        for name, handler in self.tool_registry.get_permitted_tools().items():
            if name in self.dispatcher._action_map:  # intentional: avoid overriding core actions
                continue
            self.dispatcher.register_action(name, handler)

        if loaded:
            self.trace.decision("plugins_loaded", count=len(loaded), names=loaded)

    async def _control_plane_loop(self) -> None:
        while self._running:
            try:
                if self._control_file.exists():
                    data = json.loads(self._control_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data:
                        await self._apply_control_flags(data)
                        self._control_file.write_text("{}", encoding="utf-8")
            except Exception as exc:
                self.trace.error("control_plane", str(exc))
            await asyncio.sleep(0.25)

    async def _apply_control_flags(self, flags: dict[str, Any]) -> None:
        if flags.get("interrupt"):
            self.request_interrupt(reason="dashboard")
        if flags.get("disable_actions"):
            self._failsafe_disabled = True
            self.trace.decision("failsafe_manual_disable")
        if flags.get("resume_actions"):
            self._failsafe_disabled = False
            self._failsafe_error_count = 0
            self.trace.decision("failsafe_manual_resume")

    # -- Startup / shutdown ----------------------------------------------

    async def start(self) -> None:
        self._running = True
        await self.task_manager.start()
        self._control_task = asyncio.create_task(self._control_plane_loop(), name="control_plane")
        try:
            cleaned = self.memory.cleanup_stale_data()
            self.trace.decision("memory_cleanup", **cleaned)
        except Exception:
            pass
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
                self.memory.set_preference(
                    "voice_enabled",
                    str(self._voice_on).lower(),
                    category="voice_setting",
                )
                log.info("Voice loop active")
            except Exception as exc:
                log.error(f"Voice loop failed to start: {exc} - falling back to CLI only")
                self._voice_on = False

        self._print_banner()

    async def shutdown(self) -> None:
        self._running = False
        log.info("Shutting down...")

        if self._control_task:
            self._control_task.cancel()
            try:
                await self._control_task
            except asyncio.CancelledError:
                pass

        if self._voice_loop:
            await self._voice_loop.stop()

        await self.task_manager.stop()

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
        if lower == "interrupt":
            self.request_interrupt(reason="cli")
            print("Interrupt requested.")
            return
        if lower == "failsafe on":
            self._failsafe_disabled = True
            print("Failsafe enabled.")
            return
        if lower == "failsafe off":
            self._failsafe_disabled = False
            self._failsafe_error_count = 0
            print("Failsafe disabled.")
            return
        if lower == "screen":
            await self._cmd_screen_understand()
            return
        if lower.startswith("click "):
            target = text[6:].strip()
            await self._cmd_click_target(target)
            return
        if lower.startswith("search "):
            query = text[7:].strip()
            await self._cmd_web_search(query)
            return
        if lower.startswith("serial connect"):
            args = text.split()
            port = args[2] if len(args) >= 3 else ""
            baud = args[3] if len(args) >= 4 else ""
            await self._cmd_serial_connect(port=port, baud=baud)
            return
        if lower.startswith("serial send "):
            command = text[12:].strip()
            await self._cmd_serial_send(command)
            return
        if lower == "serial disconnect":
            await self._cmd_serial_disconnect()
            return
        if lower.startswith("actuate "):
            args = text.split(maxsplit=2)
            payload = args[1:] if len(args) > 1 else []
            await self._cmd_actuate(payload)
            return
        if lower.startswith("sensor "):
            sensor = text[7:].strip()
            await self._cmd_sensor_read(sensor)
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
        if self._voice_loop is not None and hasattr(self._voice_loop, "state"):
            value = getattr(self._voice_loop.state, "value", self._voice_loop.state)
            print(f"  Voice State:  {value}")
        print(f"  Serial:       {'connected' if self.serial.is_connected else 'disconnected'}")
        print(f"  Failsafe:     {'enabled' if self._failsafe_disabled else 'normal'}")
        ok, entries, err = logger_mod.verify_audit()
        audit_status = f"OK ({entries} entries)" if ok else f"TAMPERED - {err}"
        print(f"  Audit Log:    {audit_status}")
        print(f"{'=' * 50}\n")

    def _cmd_memory(self) -> None:
        facts = self.memory.count()
        actions = len(self.memory.recent_actions(limit=10)) if hasattr(self.memory, "recent_actions") else 0
        prefs = len(self.memory.get_preferences()) if hasattr(self.memory, "get_preferences") else 0
        print(
            f"Memory summary: facts={facts}, recent_action_events={actions}, "
            f"preferences={prefs}"
        )

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
            self._failsafe_disabled = False
            self._failsafe_error_count = 0
            self.dispatcher.clear_cancel()
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

    async def _cmd_screen_understand(self) -> None:
        try:
            plan = {
                "summary": "Capture active monitor and analyze with LLaVA",
                "steps": [
                    {
                        "id": 1,
                        "action": "screen_understand",
                        "description": "Capture active monitor and analyze desktop UI",
                        "params": {},
                    }
                ],
            }
            risk = self.risk.evaluate(["screen_understand"])
            summary, _ = await self.execute_plan_with_feedback(
                intent="screen understand",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_click_target(self, target: str) -> None:
        if not target:
            print("Usage: click <target description>")
            return
        try:
            plan = {
                "summary": f"Find '{target}' on screen and click it",
                "steps": [
                    {
                        "id": 1,
                        "action": "vision_click",
                        "description": "Vision-guided UI click",
                        "params": {"target": target},
                    }
                ],
            }
            risk = self.risk.evaluate(["vision_click"])
            summary, _ = await self.execute_plan_with_feedback(
                intent=f"click {target}",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_web_search(self, query: str) -> None:
        if not query:
            print("Usage: search <query>")
            return
        try:
            plan = {
                "summary": f"Search web for: {query}",
                "steps": [
                    {
                        "id": 1,
                        "action": "web_search",
                        "description": "Fetch recent web snippets",
                        "params": {"query": query, "max_results": 5},
                    }
                ],
            }
            risk = self.risk.evaluate(["web_search"])
            summary, _ = await self.execute_plan_with_feedback(
                intent=f"search {query}",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_serial_connect(self, port: str, baud: str) -> None:
        try:
            params: dict[str, Any] = {}
            if port:
                params["port"] = port
            if baud:
                try:
                    params["baud_rate"] = int(baud)
                except ValueError:
                    print("Baud rate must be an integer.")
                    return

            plan = {
                "summary": "Connect to serial hardware",
                "steps": [
                    {
                        "id": 1,
                        "action": "serial_connect",
                        "description": "Open serial connection",
                        "params": params,
                    }
                ],
            }
            risk = self.risk.evaluate(["serial_connect"])
            summary, _ = await self.execute_plan_with_feedback(
                intent="serial connect",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_serial_send(self, command: str) -> None:
        if not command:
            print("Usage: serial send <command>")
            return
        try:
            plan = {
                "summary": "Send serial command",
                "steps": [
                    {
                        "id": 1,
                        "action": "serial_send",
                        "description": "Send raw serial command",
                        "params": {"command": command},
                    }
                ],
            }
            risk = self.risk.evaluate(["serial_send"])
            summary, _ = await self.execute_plan_with_feedback(
                intent=f"serial send {command}",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_serial_disconnect(self) -> None:
        try:
            plan = {
                "summary": "Close serial connection",
                "steps": [
                    {
                        "id": 1,
                        "action": "serial_disconnect",
                        "description": "Close serial port",
                        "params": {},
                    }
                ],
            }
            risk = self.risk.evaluate(["serial_disconnect"])
            summary, _ = await self.execute_plan_with_feedback(
                intent="serial disconnect",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_actuate(self, payload: list[str]) -> None:
        if len(payload) < 2:
            print("Usage: actuate <device> <state>")
            return
        device, state = payload[0], payload[1]
        try:
            plan = {
                "summary": f"Actuate {device} -> {state}",
                "steps": [
                    {
                        "id": 1,
                        "action": "physical_actuate",
                        "description": "Send physical actuation command via serial",
                        "params": {"device": device, "state": state},
                    }
                ],
            }
            risk = self.risk.evaluate(["physical_actuate"])
            summary, _ = await self.execute_plan_with_feedback(
                intent=f"actuate {device} {state}",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    async def _cmd_sensor_read(self, sensor: str) -> None:
        if not sensor:
            print("Usage: sensor <name>")
            return
        try:
            plan = {
                "summary": f"Read sensor {sensor}",
                "steps": [
                    {
                        "id": 1,
                        "action": "sensor_read",
                        "description": "Query a hardware sensor value via serial",
                        "params": {"sensor": sensor},
                    }
                ],
            }
            risk = self.risk.evaluate(["sensor_read"])
            summary, _ = await self.execute_plan_with_feedback(
                intent=f"sensor {sensor}",
                plan=plan,
                risk=risk,
            )
            print(f"\nJarvis: {summary}")
        finally:
            self.fsm.force_idle()

    # -- Print helpers ----------------------------------------------------

    def _print_plan(self, plan: dict, risk: Any) -> None:
        print(f"\n{'-' * 50}")
        print(f"Plan: {plan.get('summary', '(no summary)')}")
        risk_label = str(plan.get("risk_level", risk.level.name)).upper()
        confirm = bool(plan.get("confirmation_required", False))
        print(
            f"Confidence: {plan.get('confidence', 0.0):.0%} | "
            f"Risk: {risk_label} | Confirm: {'yes' if confirm else 'no'}"
        )
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
  screen            Capture active monitor and analyze with LLaVA
  click <target>    Vision-guided click on a UI target
  search <query>    Web search via DuckDuckGo
  serial connect [port] [baud]  Connect serial device
  serial send <cmd>  Send raw serial command
  serial disconnect  Close serial connection
  actuate <device> <state>  Physical actuation command via serial
  sensor <name>      Read sensor value via serial
  status            Show system status
  memory            Show memory hint
  history           Show recent audit log
  reset             Reset to IDLE
  interrupt         Cancel current execution
  failsafe on/off   Manually disable/enable action execution
  help              Show this help
  quit              Shutdown
"""
        )
