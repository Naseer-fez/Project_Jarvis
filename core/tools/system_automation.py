"""
core/tools/system_automation.py
Jarvis V3 - System Automation Tools
All tools are synchronous internally; the dispatcher awaits them via asyncio.to_thread.
"""

import os
import subprocess
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Tool Result Contract
# ─────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def to_reflection_payload(self) -> dict:
        """Normalised dict consumed by ReflectionEngine."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────
# Tool Registry  (name -> risk_score)
# ─────────────────────────────────────────────

TOOL_REGISTRY: dict[str, float] = {
    # read-only / informational
    "list_directory":    0.1,
    "read_file":         0.2,
    # state-changing / potentially destructive
    "launch_application": 0.6,
    "execute_shell":      0.7,
    "write_file":         0.8,
    "delete_file":        0.95,
}

SHELL_TIMEOUT = 10   # seconds


# ─────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────

def list_directory(path: str) -> ToolResult:
    """List contents of a directory."""
    try:
        from core.tools.builtin_tools import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=False)
        p = Path(safe_path)
        if not p.exists():
            return ToolResult(False, error=f"Path does not exist: {p}")
        if not p.is_dir():
            return ToolResult(False, error=f"Not a directory: {p}")
        entries = [
            {"name": e.name, "type": "dir" if e.is_dir() else "file", "size": e.stat().st_size if e.is_file() else None}
            for e in sorted(p.iterdir())
        ]
        lines = "\n".join(
            f"{'[DIR] ':7}{e['name']}" if e["type"] == "dir"
            else f"{'[FILE]':7}{e['name']}  ({e['size']} bytes)"
            for e in entries
        )
        return ToolResult(True, output=lines or "(empty directory)", metadata={"path": str(p), "count": len(entries)})
    except PermissionError as exc:
        return ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return ToolResult(False, error=str(exc))


def read_file(path: str, max_bytes: int = 32_768) -> ToolResult:
    """Read a text file (capped at max_bytes to protect context window)."""
    try:
        from core.tools.builtin_tools import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=False)
        p = Path(safe_path)
        if not p.is_file():
            return ToolResult(False, error=f"File not found: {p}")
        content = p.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
        truncated = len(p.read_bytes()) > max_bytes
        return ToolResult(
            True,
            output=content,
            metadata={"path": str(p), "truncated": truncated},
        )
    except PermissionError as exc:
        return ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return ToolResult(False, error=str(exc))


def write_file(path: str, content: str, overwrite: bool = False) -> ToolResult:
    """Write text content to a file. HIGH RISK – requires confirmation."""
    try:
        from core.tools.builtin_tools import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=True)
        p = Path(safe_path)
        if p.exists() and not overwrite:
            return ToolResult(False, error=f"File exists and overwrite=False: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(True, output=f"Written {len(content)} chars to {p}", metadata={"path": str(p)})
    except PermissionError as exc:
        return ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return ToolResult(False, error=str(exc))


def delete_file(path: str) -> ToolResult:
    """Delete a file. VERY HIGH RISK – requires confirmation."""
    try:
        from core.tools.builtin_tools import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=True)
        p = Path(safe_path)
        if not p.exists():
            return ToolResult(False, error=f"Path not found: {p}")
        if p.is_dir():
            return ToolResult(False, error="delete_file does not remove directories. Use a dedicated tool.")
        p.unlink()
        return ToolResult(True, output=f"Deleted: {p}", metadata={"path": str(p)})
    except PermissionError as exc:
        return ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return ToolResult(False, error=str(exc))


def launch_application(
    target: str | None = None,
    args: list[str] | None = None,
    application: str | None = None,
) -> ToolResult:
    """
    Launch a desktop application or open a file with its default handler.
    Uses os.startfile on Windows; subprocess on other platforms.
    """
    actual_target = target or application
    if not actual_target:
        return ToolResult(False, error="Either 'target' or 'application' must be provided to launch_application")
    args = args or []

    # Enforce path sandboxing on the target and arguments if they are paths
    if actual_target:
        if "/" in actual_target or "\\" in actual_target or ":" in actual_target or ".." in actual_target:
            try:
                from core.tools.builtin_tools import _assert_safe_path
                _assert_safe_path(actual_target, write_op=False)
            except Exception as e:
                return ToolResult(False, error=f"Target path escapes sandbox: {e}")

    if args:
        for arg in args:
            if "/" in arg or "\\" in arg or ":" in arg or ".." in arg:
                if arg.lower().startswith(("http://", "https://")):
                    continue
                try:
                    from core.tools.builtin_tools import _assert_safe_path
                    _assert_safe_path(arg, write_op=False)
                except Exception as e:
                    return ToolResult(False, error=f"Argument path escapes sandbox: {e}")

    try:
        if os.name == "nt":
            # os.startfile does not accept extra args, fall through to subprocess for those
            if args:
                subprocess.Popen([actual_target, *args], shell=False)
                return ToolResult(True, output=f"Launched: {actual_target} {' '.join(args)}")
            os.startfile(actual_target)  # type: ignore[attr-defined]
            return ToolResult(True, output=f"Opened: {actual_target}")
        else:
            cmd = ["xdg-open", actual_target] if not args else [actual_target, *args]
            subprocess.Popen(cmd)
            return ToolResult(True, output=f"Launched: {' '.join(cmd)}")
    except FileNotFoundError:
        return ToolResult(False, error=f"Executable/file not found: {actual_target}")
    except Exception as exc:
        return ToolResult(False, error=str(exc))


async def execute_shell(command: str, working_dir: str | None = None) -> ToolResult:
    """
    Execute a shell command and capture stdout/stderr asynchronously.
    Hard timeout of SHELL_TIMEOUT seconds – never blocks the event loop.
    Uses shell=False (shlex split) to satisfy security policy (no B602).
    """
    import shlex
    try:
        if working_dir:
            from core.tools.builtin_tools import _assert_safe_path
            try:
                _assert_safe_path(working_dir, write_op=False)
            except Exception as exc:
                return ToolResult(False, error=f"Working directory escapes sandbox: {exc}")
        cwd = Path(working_dir).expanduser().resolve() if working_dir else None
        # Split command string into a token list — avoids shell=True (B602).
        try:
            cmd_tokens = shlex.split(command, posix=(os.name != "nt"))
        except ValueError:
            return ToolResult(False, error=f"Invalid shell command syntax: {command}")

        if not cmd_tokens:
            return ToolResult(False, error="Empty command.")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_tokens,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            return ToolResult(False, error=f"Executable not found in command: {command}")

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=SHELL_TIMEOUT,
            )
            stdout = stdout_bytes.decode(errors="replace").strip()
            stderr = stderr_bytes.decode(errors="replace").strip()
            returncode = proc.returncode
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
            return ToolResult(False, error=f"Command timed out after {SHELL_TIMEOUT}s: {command}")

        success = returncode == 0
        return ToolResult(
            success,
            output=stdout,
            error=stderr,
            metadata={"returncode": returncode, "command": command},
        )
    except Exception as exc:
        return ToolResult(False, error=str(exc))


# ─────────────────────────────────────────────
# Async wrappers (all blocking calls go to thread pool)
# ─────────────────────────────────────────────

async def async_list_directory(path: str) -> ToolResult:
    return await asyncio.to_thread(list_directory, path)

async def async_read_file(path: str) -> ToolResult:
    return await asyncio.to_thread(read_file, path)

async def async_write_file(path: str, content: str, overwrite: bool = False) -> ToolResult:
    return await asyncio.to_thread(write_file, path, content, overwrite)

async def async_delete_file(path: str) -> ToolResult:
    return await asyncio.to_thread(delete_file, path)

async def async_launch_application(
    target: str | None = None,
    args: list[str] | None = None,
    application: str | None = None,
) -> ToolResult:
    return await asyncio.to_thread(launch_application, target=target, args=args, application=application)

async def async_execute_shell(command: str, working_dir: str | None = None) -> ToolResult:
    return await execute_shell(command, working_dir)

