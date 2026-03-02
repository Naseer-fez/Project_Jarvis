"""Execution dispatcher with core-tool and dynamic-integration routing."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from core.agentic.autonomy_policy import AutonomyPolicy, PolicyDecision, PolicyVerdict
from core.tools.system_automation import (
    TOOL_REGISTRY,
    ToolResult,
    async_delete_file,
    async_execute_shell,
    async_launch_application,
    async_list_directory,
    async_read_file,
    async_write_file,
)

try:
    from integrations.registry import integration_registry
except Exception:  # noqa: BLE001
    integration_registry = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CONFIRM_THRESHOLD = 0.5

INTEGRATION_RISK_REGISTRY: dict[str, float] = {
    "read_emails": 0.1,
    "search_emails": 0.1,
    "list_events": 0.1,
    "get_current_weather": 0.1,
    "send_email": 0.8,
    "send_whatsapp": 0.8,
    "add_event": 0.6,
}


class DispatchError(Exception):
    pass


class Dispatcher:
    def __init__(self, autonomy_policy: AutonomyPolicy, reflection_engine, voice_layer=None):
        self.policy = autonomy_policy
        self.reflection = reflection_engine
        self.voice = voice_layer

        # Rate limiting — 30 tool calls per 60-second window per instance
        self._call_count: int = 0
        self._call_window_start: float = time.time()
        self.MAX_CALLS_PER_MINUTE: int = 30

        self._core_tools = {
            "list_directory": self._run_list_directory,
            "read_file": self._run_read_file,
            "write_file": self._run_write_file,
            "delete_file": self._run_delete_file,
            "launch_application": self._run_launch_application,
            "execute_shell": self._run_execute_shell,
        }

    def _sanitize_args(self, args: dict) -> dict:
        """Strip null bytes and truncate oversized string values."""
        sanitized = {}
        for key, val in args.items():
            if isinstance(val, str):
                val = val.replace("\x00", "")   # strip null bytes
                if len(val) > 4096:
                    val = val[:4096]            # truncate oversized
            sanitized[key] = val
        return sanitized

    async def dispatch(self, action: dict[str, Any]) -> ToolResult:
        tool_name = str(action.get("tool", "")).strip()
        args = action.get("args", {}) or {}
        rationale = str(action.get("rationale", "")).strip()

        # Rate limiting — reset window if 60 s have elapsed
        now = time.time()
        if now - self._call_window_start > 60:
            self._call_count = 0
            self._call_window_start = now
        self._call_count += 1
        if self._call_count > self.MAX_CALLS_PER_MINUTE:
            return ToolResult(False, error="Rate limit exceeded: 30 tool calls/minute")

        # Sanitize args before dispatching
        args = self._sanitize_args(args)

        # 1) Built-in core tool registry first.
        if tool_name in TOOL_REGISTRY and tool_name in self._core_tools:
            risk_score = float(TOOL_REGISTRY.get(tool_name, 1.0))
            logger.info("Dispatch core tool='%s' risk=%.2f rationale='%s'", tool_name, risk_score, rationale)
            return await self._dispatch_core_tool(tool_name, args, risk_score)

        # 2) Dynamic integrations second.
        if self._has_integration_tool(tool_name):
            risk_score = float(INTEGRATION_RISK_REGISTRY.get(tool_name, 0.6))
            logger.info(
                "Dispatch integration tool='%s' risk=%.2f rationale='%s'",
                tool_name,
                risk_score,
                rationale,
            )
            return await self._dispatch_integration_tool(tool_name, args, risk_score)

        result = ToolResult(False, error=f"Unknown tool: '{tool_name}'")
        await self._feed_reflection(tool_name, args, result)
        return result

    def _integration_tool_names(self) -> set[str]:
        if integration_registry is None:
            return set()

        try:
            return {
                str(tool.get("name", "")).strip()
                for tool in integration_registry.get_tools()
                if isinstance(tool, dict)
            }
        except Exception:  # noqa: BLE001
            return set()

    def _has_integration_tool(self, tool_name: str) -> bool:
        if integration_registry is None:
            return False

        getter = getattr(integration_registry, "get_tool", None)
        if callable(getter):
            try:
                return getter(tool_name) is not None
            except Exception:  # noqa: BLE001
                pass

        return tool_name in self._integration_tool_names()

    async def _dispatch_core_tool(self, tool_name: str, args: dict[str, Any], risk_score: float) -> ToolResult:
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by policy or user rejection")
            await self._feed_reflection(tool_name, args, result)
            return result

        try:
            result = await self._core_tools[tool_name](args)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Core tool '%s' failed", tool_name)
            result = ToolResult(False, error=f"Dispatcher error: {exc}")

        await self._feed_reflection(tool_name, args, result)
        return result

    async def _dispatch_integration_tool(self, tool_name: str, args: dict[str, Any], risk_score: float) -> ToolResult:
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by policy or user rejection")
            await self._feed_reflection(tool_name, args, result)
            return result

        try:
            payload = await integration_registry.execute(tool_name, args)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            payload = {"success": False, "data": None, "error": str(exc)}

        result = ToolResult(
            success=bool(payload.get("success", False)),
            output=str(payload.get("data", "")) if payload.get("data") is not None else "",
            error=str(payload.get("error", "") or ""),
            metadata={"integration": True, "tool": tool_name},
        )

        await self._feed_reflection(tool_name, args, result)
        return result

    async def _check_policy(self, tool_name: str, args: dict[str, Any], risk_score: float) -> bool:
        class Context:
            def __init__(self, score: float) -> None:
                self.interrupt_flag = False
                self.paused = False
                self.risk_score = score
                self.confidence_score = 0.99

            def snapshot(self) -> dict[str, float]:
                return {"risk_score": self.risk_score}

        context = Context(risk_score)
        decision: PolicyDecision = self.policy.check(context=context, action_name=tool_name, params=args)

        if decision.verdict == PolicyVerdict.DENY:
            logger.warning("Policy denied tool '%s': %s", tool_name, decision.reason)
            return False

        if decision.verdict == PolicyVerdict.ALLOW and risk_score < CONFIRM_THRESHOLD:
            return True

        return await self._ask_voice_confirm(tool_name, args)

    async def _ask_voice_confirm(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self.voice is None:
            logger.warning("No voice layer attached; blocking high-risk action '%s'", tool_name)
            return False

        summary = _summarise_action(tool_name, args)
        prompt = f"I need approval before I {summary}. Proceed?"
        try:
            return bool(await self.voice.ask_confirm(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Voice confirmation failed for '%s': %s", tool_name, exc)
            return False

    async def _feed_reflection(self, tool_name: str, args: dict[str, Any], result: ToolResult) -> None:
        if self.reflection is None:
            return
        payload = {
            "tool": tool_name,
            "args": args,
            **result.to_reflection_payload(),
        }
        try:
            fn = getattr(self.reflection, "record_action", None)
            if fn is None:
                return
            if asyncio.iscoroutinefunction(fn):
                await fn(payload)
            else:
                await asyncio.to_thread(fn, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Reflection hook failed: %s", exc)

    async def _run_list_directory(self, args: dict[str, Any]) -> ToolResult:
        return await async_list_directory(args["path"])

    async def _run_read_file(self, args: dict[str, Any]) -> ToolResult:
        return await async_read_file(args["path"])

    async def _run_write_file(self, args: dict[str, Any]) -> ToolResult:
        return await async_write_file(args["path"], args["content"], bool(args.get("overwrite", False)))

    async def _run_delete_file(self, args: dict[str, Any]) -> ToolResult:
        return await async_delete_file(args["path"])

    async def _run_launch_application(self, args: dict[str, Any]) -> ToolResult:
        return await async_launch_application(args["target"], args.get("args"))

    async def _run_execute_shell(self, args: dict[str, Any]) -> ToolResult:
        return await async_execute_shell(args["command"], args.get("working_dir"))


def _summarise_action(tool_name: str, args: dict[str, Any]) -> str:
    summaries = {
        "execute_shell": lambda x: f"run shell command '{x.get('command', '')}'",
        "write_file": lambda x: f"write file '{x.get('path', '')}'",
        "delete_file": lambda x: f"delete file '{x.get('path', '')}'",
        "launch_application": lambda x: f"launch '{x.get('target', '')}'",
        "send_email": lambda x: f"send an email to {x.get('to', '')}",
        "send_whatsapp": lambda x: f"send a WhatsApp message to {x.get('to', '')}",
        "add_event": lambda x: f"add calendar event '{x.get('title', '')}'",
    }
    return summaries.get(tool_name, lambda _: f"run '{tool_name}'")(args)


class _StepResult:
    """Lightweight result object returned by ToolDispatcher.execute_plan()."""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str | None = None,
        timed_out: bool = False,
    ) -> None:
        self.success = success
        self.output = output
        self.error = error
        self.timed_out = timed_out

    def __repr__(self) -> str:
        return f"<StepResult success={self.success} output={self.output!r} error={self.error!r}>"


class ToolDispatcher:
    """
    Synchronous tool dispatcher used by legacy (phase3/4/5) test files.

    Provides:
      * execute_plan(plan) — runs steps synchronously with sandbox + rollback
      * register_action(name, fn) — register custom action handlers
      * _capture_screen_image / _gui_click — stubbed for monkeypatching in tests
    """

    def __init__(
        self,
        config=None,
        memory=None,
        vision=None,
        serial_controller=None,
    ) -> None:
        import configparser as _cp

        self.config = config
        self.memory = memory
        self.vision = vision
        self.serial_controller = serial_controller

        # Parse execution settings from config
        if isinstance(config, _cp.ConfigParser):
            ex = config["execution"] if config.has_section("execution") else {}
            raw_dirs = ex.get("safe_directories", "")
            self._safe_dirs = [
                str(Path(d.strip()).resolve())
                for d in raw_dirs.split(",")
                if d.strip()
            ]
            self._max_read_bytes = int(ex.get("max_read_bytes", 1_000_000))
            self._allow_gui = ex.get("allow_gui_automation", "false").lower() == "true"
            self._rollback = ex.get("rollback_on_failure", "false").lower() == "true"
            self._step_timeout = float(ex.get("step_timeout_s", 30))
        else:
            self._safe_dirs = []
            self._max_read_bytes = 1_000_000
            self._allow_gui = False
            self._rollback = False
            self._step_timeout = 30.0

        self._custom_actions: dict[str, Any] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def register_action(self, name: str, fn) -> None:
        """Register a custom action handler callable(params) -> str."""
        self._custom_actions[name] = fn

    def execute_plan(self, plan: dict[str, Any]) -> list[_StepResult]:
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        results: list[_StepResult] = []
        rollback_files: list[Path] = []

        for step in steps:
            action = str(step.get("action", "")).strip()
            params = step.get("params", {}) or {}

            result = self._execute_step(action, params)
            results.append(result)

            if result.success and action in ("file_write", "write_file"):
                # Track written files for potential rollback
                p = params.get("path", "")
                if p:
                    safe = self._resolve_safe(p)
                    if safe:
                        rollback_files.append(safe)

            if not result.success and self._rollback:
                # Roll back successfully written files
                for f in rollback_files:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass
                break

        return results

    # ── Step dispatch ─────────────────────────────────────────────────────

    def _execute_step(self, action: str, params: dict[str, Any]) -> _StepResult:
        import threading

        result_holder: list[_StepResult] = []
        exc_holder: list[BaseException] = []

        def _run():
            try:
                r = self._dispatch_action(action, params)
                result_holder.append(r)
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self._step_timeout)

        if t.is_alive():
            return _StepResult(False, error=f"Step '{action}' timed out", timed_out=True)
        if exc_holder:
            return _StepResult(False, error=str(exc_holder[0]))
        return result_holder[0] if result_holder else _StepResult(False, error="No result")

    def _dispatch_action(self, action: str, params: dict[str, Any]) -> _StepResult:
        # Custom registered actions first
        if action in self._custom_actions:
            out = self._custom_actions[action](params)
            return _StepResult(True, output=str(out) if out is not None else "")

        if action in ("file_read", "read_file"):
            return self._action_file_read(params)
        if action in ("file_write", "write_file"):
            return self._action_file_write(params)
        if action in ("screen_understand",):
            return self._action_screen_understand(params)
        if action in ("vision_click",):
            return self._action_vision_click(params)
        if action == "physical_actuate":
            return self._action_physical_actuate(params)
        if action == "sensor_read":
            return self._action_sensor_read(params)

        return _StepResult(False, error=f"Unknown action: '{action}'")

    def _action_physical_actuate(self, params: dict[str, Any]) -> _StepResult:
        if not self.serial_controller:
            return _StepResult(False, error="No serial controller configured")
        device = str(params.get("device", "")).upper()
        state = str(params.get("state", "")).upper()
        command = f"{device}:{state}"
        try:
            out = self.serial_controller.send(command)
            return _StepResult(True, output=str(out))
        except Exception as e:
            return _StepResult(False, error=str(e))

    def _action_sensor_read(self, params: dict[str, Any]) -> _StepResult:
        if not self.serial_controller:
            return _StepResult(False, error="No serial controller configured")
        sensor = str(params.get("sensor", "")).upper()
        command = f"READ:{sensor}"
        try:
            out = self.serial_controller.send(command)
            return _StepResult(True, output=str(out))
        except Exception as e:
            return _StepResult(False, error=str(e))

    # ── Built-in actions ──────────────────────────────────────────────────

    def _resolve_safe(self, path_str: str) -> "Path | None":
        """Resolve path to an absolute Path if it is within a safe directory."""
        if not self._safe_dirs:
            return Path(path_str).resolve()

        # Relative paths are resolved relative to the first safe dir
        p = Path(path_str)
        if not p.is_absolute():
            p = Path(self._safe_dirs[0]) / p

        resolved = p.resolve()
        for safe in self._safe_dirs:
            try:
                resolved.relative_to(safe)
                return resolved
            except ValueError:
                continue
        return None  # outside sandbox

    def _action_file_read(self, params: dict[str, Any]) -> _StepResult:
        path_str = str(params.get("path", ""))
        if not path_str:
            return _StepResult(False, error="file_read: 'path' parameter required")

        resolved = self._resolve_safe(path_str)
        if resolved is None:
            return _StepResult(False, error=f"Path '{path_str}' is outside safe directories")

        if not resolved.exists():
            return _StepResult(False, error=f"File not found: {resolved}")

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            if self._max_read_bytes and len(content.encode()) > self._max_read_bytes:
                content = content[: self._max_read_bytes]
            return _StepResult(True, output=content)
        except Exception as exc:  # noqa: BLE001
            return _StepResult(False, error=str(exc))

    def _action_file_write(self, params: dict[str, Any]) -> _StepResult:
        path_str = str(params.get("path", ""))
        content = str(params.get("content", ""))
        if not path_str:
            return _StepResult(False, error="file_write: 'path' parameter required")

        resolved = self._resolve_safe(path_str)
        if resolved is None:
            return _StepResult(False, error=f"Path '{path_str}' is outside safe directories")

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return _StepResult(True, output=f"Written {len(content)} chars to {resolved}")
        except Exception as exc:  # noqa: BLE001
            return _StepResult(False, error=str(exc))

    def _action_screen_understand(self, params: dict[str, Any]) -> _StepResult:
        capture_mode = params.get("capture_mode", "active_monitor")
        output_path = Path("screen_capture.png")
        try:
            img_path, offset = self._capture_screen_image(output_path, capture_mode)
        except Exception as exc:  # noqa: BLE001
            return _StepResult(False, error=f"Screen capture failed: {exc}")

        if self.vision is None:
            return _StepResult(False, error="No vision backend available")

        try:
            description = self.vision.analyze(str(img_path))
            return _StepResult(True, output=description)
        except Exception as exc:  # noqa: BLE001
            return _StepResult(False, error=f"Vision analysis failed: {exc}")

    def _action_vision_click(self, params: dict[str, Any]) -> _StepResult:
        if not self._allow_gui:
            return _StepResult(False, error="GUI automation disabled by config")

        target = str(params.get("target", ""))
        dry_run = bool(params.get("dry_run", False))
        capture_mode = params.get("capture_mode", "active_monitor")
        output_path = Path("vision_click_capture.png")

        try:
            img_path, offset = self._capture_screen_image(output_path, capture_mode)
        except Exception as exc:  # noqa: BLE001
            return _StepResult(False, error=f"Screen capture failed: {exc}")

        if self.vision is None:
            return _StepResult(False, error="No vision backend available")

        prompt = f"Find '{target}'. Return JSON: {{x, y, confidence, reason, not_found}}"
        try:
            raw = self.vision.analyze(str(img_path), prompt)
            import json as _json
            data = _json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            return _StepResult(False, error=f"Vision response parse failed: {exc}")

        if data.get("not_found"):
            return _StepResult(False, error=f"Target '{target}' not found on screen")

        rel_x, rel_y = int(data.get("x", 0)), int(data.get("y", 0))
        ox, oy = offset if offset else (0, 0)
        screen_x, screen_y = rel_x + ox, rel_y + oy
        out = f"Found '{target}' at rel=({rel_x},{rel_y}) screen=({screen_x},{screen_y})"

        if not dry_run:
            self._gui_click({"x": screen_x, "y": screen_y})

        return _StepResult(True, output=out)

    # ── Hookable helpers (monkeypatched in tests) ─────────────────────────

    def _capture_screen_image(self, output_path: Path, capture_mode: str) -> tuple[Path, tuple[int, int]]:
        """Capture screen. Returns (image_path, (offset_x, offset_y)). Override in tests."""
        try:
            import PIL.ImageGrab as _ig  # type: ignore[import]
            img = _ig.grab()
            img.save(str(output_path))
            return output_path, (0, 0)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Screen capture unavailable: {exc}") from exc

    def _gui_click(self, params: dict[str, Any]) -> str:
        """Perform a mouse click. Override in tests."""
        try:
            import pyautogui  # type: ignore[import]
            pyautogui.click(params.get("x", 0), params.get("y", 0))
            return "clicked"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"GUI click failed: {exc}") from exc


__all__ = ["DispatchError", "Dispatcher", "ToolDispatcher"]
