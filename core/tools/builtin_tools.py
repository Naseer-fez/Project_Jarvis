"""
Built-in tools for Jarvis.
All tools are async coroutines and sandboxed to allowed directories.
"""

import asyncio
import gzip
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Jarvis.Tools")

ALLOWED_DIRECTORIES = [
    Path("./workspace").resolve(),
    Path("./outputs").resolve(),
]

# Project root sandbox — all resolved paths must stay inside here.
_SANDBOX_ROOT = Path("D:/AI/Jarvis").resolve()


def _assert_safe_path(path_str: str) -> Path:
    """Raise PermissionError / ValueError if path is outside the sandbox."""
    # Block path traversal sequences before resolution
    if ".." in str(Path(path_str)):
        raise PermissionError(f"Path traversal blocked: {path_str}")

    resolved = Path(path_str).resolve()
    sandbox = _SANDBOX_ROOT

    # Must be inside project sandbox
    if not str(resolved).startswith(str(sandbox)):
        raise PermissionError(f"Path outside sandbox: {resolved}")

    # Symlink must not escape sandbox
    if resolved.is_symlink():
        link_target = resolved.resolve()
        if not str(link_target).startswith(str(sandbox)):
            raise PermissionError(f"Symlink escapes sandbox: {link_target}")

    # Also check legacy ALLOWED_DIRECTORIES for backward compatibility
    target = resolved
    for allowed in ALLOWED_DIRECTORIES:
        try:
            target.relative_to(allowed)
            return target
        except ValueError:
            continue
    raise ValueError(f"Path '{path_str}' is outside the sandbox. Allowed: {[str(d) for d in ALLOWED_DIRECTORIES]}")


# ── System tools ────────────────────────────────────────────────────────────

async def get_time() -> str:
    """Returns current local time and date."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("Current time: %H:%M:%S on %A, %B %d, %Y")


async def get_system_stats() -> str:
    """Returns basic system resource usage."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return (
            f"CPU: {cpu}% | "
            f"Memory: {mem.percent}% used ({mem.available // 1024 // 1024} MB free) | "
            f"Disk: {disk.percent}% used ({disk.free // 1024 // 1024 // 1024} GB free)"
        )
    except ImportError:
        return f"Platform: {platform.system()} {platform.release()} | (install psutil for detailed stats)"


# ── File tools ──────────────────────────────────────────────────────────────

async def list_directory(path: str = "./workspace") -> str:
    """Lists files in a sandboxed directory."""
    safe = _assert_safe_path(path)
    if not safe.exists():
        return f"Directory '{path}' does not exist."
    entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for e in entries:
        tag = "[DIR] " if e.is_dir() else "[FILE]"
        size = f" ({e.stat().st_size} bytes)" if e.is_file() else ""
        lines.append(f"{tag} {e.name}{size}")
    return "\n".join(lines) if lines else "(empty directory)"


async def read_file(path: str) -> str:
    """Reads a text file from the sandbox."""
    safe = _assert_safe_path(path)
    if not safe.exists():
        return f"File '{path}' not found."
    if not safe.is_file():
        return f"'{path}' is not a file."
    size = os.path.getsize(safe)
    if size > 10 * 1024 * 1024:   # 10 MB hard limit
        raise ValueError(f"File too large: {size} bytes (max 10MB)")
    if size > 100_000:
        return f"File too large ({size} bytes). Max 100KB."
    return safe.read_text(encoding="utf-8", errors="replace")


async def write_file_safe(path: str, content: str) -> str:
    """Writes content to a file in the sandbox (creates if needed)."""
    safe = _assert_safe_path(path)
    safe.parent.mkdir(parents=True, exist_ok=True)
    safe.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to '{path}'."


# ── Memory tools ─────────────────────────────────────────────────────────────

_memory_store: list[dict] = []  # In-process simple memory


async def search_memory(query: str, limit: int = 5) -> str:
    """Simple keyword search over in-session memory."""
    query_lower = query.lower()
    matches = [
        m for m in _memory_store
        if query_lower in m.get("content", "").lower()
    ]
    if not matches:
        return f"No memory entries found matching '{query}'."
    results = matches[-limit:]
    return "\n".join(
        f"[{m.get('timestamp', '?')}] {m['content']}" for m in results
    )


async def log_event(content: str, category: str = "general") -> str:
    """Logs an event to in-session memory and the outputs log file."""
    import datetime
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "category": category,
        "content": content,
    }
    _memory_store.append(entry)

    log_path = Path("./outputs/memory_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return f"Event logged: [{category}] {content}"


def register_all_tools(router):
    """Register all built-in tools with a ToolRouter instance."""
    # ── Core tools ─────────────────────────────────────────────────────────
    router.register("get_time", get_time)
    router.register("get_system_stats", get_system_stats)
    router.register("list_directory", list_directory)
    router.register("read_file", read_file)
    router.register("write_file_safe", write_file_safe)
    router.register("search_memory", search_memory)
    router.register("log_event", log_event)

    # ── Hardware tools (Session 7) ─────────────────────────────────────────
    try:
        from core.tools.hardware_tools import (
            send_hardware_command,
            read_sensor,
            list_hardware_devices,
            ping_device,
        )
        router.register("send_hardware_command", send_hardware_command)
        router.register("read_sensor", read_sensor)
        router.register("list_hardware_devices", list_hardware_devices)
        router.register("ping_device", ping_device)
        logger.info("Hardware tools registered (Session 7)")
    except Exception as e:
        logger.warning("Hardware tools unavailable: %s", e)

    # ── Screen tools (Session 7) ───────────────────────────────────────────
    try:
        from core.tools.screen import (
            capture_screen,
            capture_region,
            find_text_on_screen,
            describe_screen,
        )
        router.register("capture_screen", capture_screen)
        router.register("capture_region", capture_region)
        router.register("find_text_on_screen", find_text_on_screen)
        router.register("describe_screen", describe_screen)
        logger.info("Screen tools registered (Session 7)")
    except Exception as e:
        logger.warning("Screen tools unavailable: %s", e)

    # ── GUI control tools (Session 7) ──────────────────────────────────────
    try:
        from core.tools.gui_control import (
            click,
            double_click,
            right_click,
            type_text,
            hotkey,
            get_active_window,
        )
        router.register("click", click)
        router.register("double_click", double_click)
        router.register("right_click", right_click)
        router.register("type_text", type_text)
        router.register("hotkey", hotkey)
        router.register("get_active_window", get_active_window)
        logger.info("GUI control tools registered (Session 7)")
    except Exception as e:
        logger.warning("GUI control tools unavailable: %s", e)

    logger.info("Registered %d tools total: %s", len(router.registered_tools()), router.registered_tools())


