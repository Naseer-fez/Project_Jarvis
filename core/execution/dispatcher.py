"""
core/execution/dispatcher.py - Secure plan step dispatcher.

Maps planner actions to explicit Python handlers. No dynamic eval/exec.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class DispatchResult:
    step_id: int
    action: str
    success: bool
    output: str
    error: str | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }


class ToolDispatcher:
    def __init__(self, config, memory=None, vision=None, serial_controller=None) -> None:
        self._config = config
        self._memory = memory
        self._vision = vision
        self._serial = serial_controller

        self._safe_dirs = self._load_safe_dirs()
        self._default_root = self._safe_dirs[0]
        self._max_read_bytes = int(
            config.get("execution", "max_read_bytes", fallback="200000")
        )
        self._allowed_apps = self._load_allowed_apps()

        self._action_map: dict[str, Callable[[dict[str, Any]], str]] = {
            # Memory
            "memory_read": self._memory_read,
            "memory_write": self._memory_write,
            "recall": self._memory_read,
            "store_fact": self._memory_write,

            # User-visible output actions
            "speak": self._echo,
            "display": self._echo,
            "status": self._status,
            "health_check": self._health_check,

            # File/system
            "file_read": self._file_read,
            "read_file": self._file_read,
            "file_write": self._file_write,
            "write_file": self._file_write,
            "system_stats": self._system_stats,
            "app_open": self._app_open,
            "process_spawn": self._app_open,

            # Vision/screen/gui
            "vision_analyze": self._vision_analyze,
            "screen_capture": self._screen_capture,
            "gui_click": self._gui_click,
            "gui_type": self._gui_type,
            "gui_hotkey": self._gui_hotkey,

            # Hardware
            "serial_connect": self._serial_connect,
            "serial_send": self._serial_send,
            "serial_disconnect": self._serial_disconnect,

            # Optional real-time data
            "web_search": self._web_search,
        }

    def execute_plan(self, plan: dict[str, Any]) -> list[DispatchResult]:
        """
        Execute each plan step in order and return per-step results.
        """
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            return [
                DispatchResult(
                    step_id=0,
                    action="invalid_plan",
                    success=False,
                    output="",
                    error="Plan 'steps' must be a list.",
                )
            ]

        results: list[DispatchResult] = []
        for idx, step in enumerate(steps, start=1):
            step_id = idx
            action = "unknown"
            params: dict[str, Any] = {}
            if isinstance(step, dict):
                step_id = int(step.get("id", idx))
                action = str(step.get("action", "unknown")).strip().lower()
                raw_params = step.get("params", {})
                if isinstance(raw_params, dict):
                    params = raw_params

            t0 = time.monotonic()
            handler = self._action_map.get(action)
            if handler is None:
                results.append(
                    DispatchResult(
                        step_id=step_id,
                        action=action,
                        success=False,
                        output="",
                        error=f"Action '{action}' is not mapped to a dispatcher handler.",
                        duration_s=time.monotonic() - t0,
                    )
                )
                continue

            try:
                output = handler(params)
                results.append(
                    DispatchResult(
                        step_id=step_id,
                        action=action,
                        success=True,
                        output=str(output),
                        duration_s=time.monotonic() - t0,
                    )
                )
            except Exception as exc:
                results.append(
                    DispatchResult(
                        step_id=step_id,
                        action=action,
                        success=False,
                        output="",
                        error=str(exc),
                        duration_s=time.monotonic() - t0,
                    )
                )

        return results

    # -- Security helpers -------------------------------------------------

    def _load_safe_dirs(self) -> list[Path]:
        raw = self._config.get(
            "execution", "safe_directories", fallback="workspace,outputs,data"
        )
        dirs = [d.strip() for d in raw.split(",") if d.strip()]
        resolved: list[Path] = []
        for entry in dirs:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            else:
                path = path.resolve()
            path.mkdir(parents=True, exist_ok=True)
            resolved.append(path)

        if not resolved:
            fallback = (Path.cwd() / "workspace").resolve()
            fallback.mkdir(parents=True, exist_ok=True)
            resolved = [fallback]
        return resolved

    def _load_allowed_apps(self) -> set[str]:
        raw = self._config.get("execution", "allowed_apps", fallback="notepad,calc")
        return {a.strip().lower() for a in raw.split(",") if a.strip()}

    def _resolve_safe_path(self, raw_path: str, for_write: bool = False) -> Path:
        if not raw_path:
            raise ValueError("Missing required 'path' parameter.")

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self._default_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        for root in self._safe_dirs:
            try:
                candidate.relative_to(root)
                if for_write:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                return candidate
            except ValueError:
                continue

        raise PermissionError(
            f"Path '{candidate}' is outside safe directories: "
            f"{', '.join(str(d) for d in self._safe_dirs)}"
        )

    # -- Action handlers --------------------------------------------------

    def _echo(self, params: dict[str, Any]) -> str:
        return str(params.get("text", "")).strip()

    def _status(self, params: dict[str, Any]) -> str:
        del params
        return "Dispatcher online. Ready for mapped tool execution."

    def _health_check(self, params: dict[str, Any]) -> str:
        del params
        return self._system_stats({})

    def _memory_read(self, params: dict[str, Any]) -> str:
        if self._memory is None:
            return "Memory subsystem unavailable."
        query = str(params.get("query", params.get("key", ""))).strip()
        if not query:
            raise ValueError("memory_read requires 'query' or 'key'.")
        return str(self._memory.recall(query))

    def _memory_write(self, params: dict[str, Any]) -> str:
        if self._memory is None:
            return "Memory subsystem unavailable."
        key = str(params.get("key", "")).strip()
        value = str(params.get("value", "")).strip()
        if not key or not value:
            raise ValueError("memory_write/store_fact requires 'key' and 'value'.")
        self._memory.store_fact(
            key,
            value,
            source="dispatcher",
            metadata={"action": "memory_write"},
        )
        return f"Stored memory fact: {key}"

    def _file_read(self, params: dict[str, Any]) -> str:
        path = self._resolve_safe_path(str(params.get("path", "")), for_write=False)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        size = path.stat().st_size
        if size > self._max_read_bytes:
            raise ValueError(
                f"File exceeds read limit ({size} > {self._max_read_bytes} bytes)."
            )
        return path.read_text(encoding="utf-8", errors="replace")

    def _file_write(self, params: dict[str, Any]) -> str:
        path = self._resolve_safe_path(str(params.get("path", "")), for_write=True)
        content = str(params.get("content", ""))
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    def _system_stats(self, params: dict[str, Any]) -> str:
        del params
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            return (
                f"CPU {cpu:.1f}% | Memory {mem.percent:.1f}% used "
                f"({mem.available // (1024 * 1024)} MB free)"
            )
        except Exception:
            return f"Platform: {platform.system()} {platform.release()}"

    def _app_open(self, params: dict[str, Any]) -> str:
        app = str(params.get("app", "")).strip().lower()
        if not app:
            raise ValueError("app_open requires 'app'.")
        if app not in self._allowed_apps:
            raise PermissionError(
                f"Application '{app}' is not in allowlist: {sorted(self._allowed_apps)}"
            )

        args = params.get("args", [])
        if args is None:
            args = []
        if not isinstance(args, list):
            raise ValueError("'args' must be a list of strings.")
        cmd = [app] + [str(a) for a in args]
        proc = subprocess.Popen(cmd)
        return f"Started '{app}' with pid={proc.pid}"

    def _vision_analyze(self, params: dict[str, Any]) -> str:
        if self._vision is None:
            return "Vision tool unavailable."
        path = str(params.get("path", "")).strip()
        prompt = str(params.get("prompt", "Describe this image.")).strip()
        if not path:
            raise ValueError("vision_analyze requires 'path'.")
        return str(self._vision.analyze(path, prompt=prompt))

    def _screen_capture(self, params: dict[str, Any]) -> str:
        output_path = str(params.get("path", "")).strip()
        if not output_path:
            output_path = f"screenshot_{int(time.time())}.png"
        path = self._resolve_safe_path(output_path, for_write=True)

        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise RuntimeError(
                "screen_capture requires pillow (PIL.ImageGrab)."
            ) from exc

        image = ImageGrab.grab()
        image.save(path)
        return f"Screenshot saved: {path}"

    def _gui_click(self, params: dict[str, Any]) -> str:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError("gui_click requires pyautogui.") from exc

        x = params.get("x")
        y = params.get("y")
        button = str(params.get("button", "left"))
        clicks = int(params.get("clicks", 1))
        if x is None or y is None:
            pyautogui.click(button=button, clicks=clicks)
        else:
            pyautogui.click(x=int(x), y=int(y), button=button, clicks=clicks)
        return "GUI click executed."

    def _gui_type(self, params: dict[str, Any]) -> str:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError("gui_type requires pyautogui.") from exc

        text = str(params.get("text", ""))
        interval = float(params.get("interval", 0.02))
        if not text:
            raise ValueError("gui_type requires 'text'.")
        pyautogui.write(text, interval=interval)
        return "GUI typing executed."

    def _gui_hotkey(self, params: dict[str, Any]) -> str:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError("gui_hotkey requires pyautogui.") from exc

        keys = params.get("keys", [])
        if not isinstance(keys, list) or not keys:
            raise ValueError("gui_hotkey requires non-empty list 'keys'.")
        pyautogui.hotkey(*[str(k) for k in keys])
        return "GUI hotkey executed."

    def _serial_connect(self, params: dict[str, Any]) -> str:
        if self._serial is None:
            raise RuntimeError("Serial controller unavailable.")
        port = str(params.get("port", "")).strip()
        if not port:
            raise ValueError("serial_connect requires 'port'.")
        baud = int(params.get("baud_rate", 9600))
        self._serial.connect(port=port, baud_rate=baud)
        return f"Serial connected: {port} @ {baud}"

    def _serial_send(self, params: dict[str, Any]) -> str:
        if self._serial is None:
            raise RuntimeError("Serial controller unavailable.")
        command = str(params.get("command", "")).strip()
        if not command:
            raise ValueError("serial_send requires 'command'.")
        reply = self._serial.send(command)
        return f"Serial command sent. Reply: {reply}"

    def _serial_disconnect(self, params: dict[str, Any]) -> str:
        del params
        if self._serial is None:
            raise RuntimeError("Serial controller unavailable.")
        self._serial.disconnect()
        return "Serial disconnected."

    def _web_search(self, params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            raise ValueError("web_search requires 'query'.")

        max_results = int(params.get("max_results", 3))
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "web_search requires duckduckgo-search package."
            ) from exc

        lines: list[str] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    title = item.get("title", "").strip()
                    href = item.get("href", "").strip()
                    body = item.get("body", "").strip()
                    lines.append(f"- {title} | {href}\n  {body}")
        except Exception as exc:
            return f"Web search error: {exc}"
        return "\n".join(lines) if lines else "No web results found."
