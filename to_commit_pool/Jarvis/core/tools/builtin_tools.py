"""
Built-in tools for Jarvis.
All tools are async coroutines and sandboxed to allowed directories.
"""

import json
import logging
import os
import platform
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger("Jarvis.Tools")

ALLOWED_DIRECTORIES = [
    (_PROJECT_ROOT / "workspace").resolve(),
    (_PROJECT_ROOT / "outputs").resolve(),
]

# Project root sandbox — all resolved paths must stay inside here.
_SANDBOX_ROOT = _PROJECT_ROOT


def _assert_safe_path(path_str: str, write_op: bool = False) -> Path:
    """Raise PermissionError / ValueError if path is outside the sandbox."""
    # Block path traversal sequences before resolution
    if ".." in str(Path(path_str)):
        raise PermissionError(f"Path traversal blocked: {path_str}")

    resolved = Path(path_str).resolve()
    sandbox = _SANDBOX_ROOT

    resolved_str = str(resolved)
    sandbox_str = str(sandbox)
    if os.name == "nt":
        resolved_str = resolved_str.lower()
        sandbox_str = sandbox_str.lower()

    # Must be inside project sandbox
    if not resolved_str.startswith(sandbox_str):
        raise PermissionError(f"Path outside sandbox: {resolved}")

    # Symlink must not escape sandbox
    if resolved.is_symlink():
        link_target = resolved.resolve()
        link_target_str = str(link_target)
        if os.name == "nt":
            link_target_str = link_target_str.lower()
        if not link_target_str.startswith(sandbox_str):
            raise PermissionError(f"Symlink escapes sandbox: {link_target}")

    # Also check legacy ALLOWED_DIRECTORIES for backward compatibility
    target = resolved
    if write_op:
        allowed_dirs = ALLOWED_DIRECTORIES
    else:
        allowed_dirs = ALLOWED_DIRECTORIES + [
            (_PROJECT_ROOT / "config").resolve(),
            (_PROJECT_ROOT / "data").resolve(),
            (_PROJECT_ROOT / "logs").resolve(),
            (_PROJECT_ROOT / "core").resolve(),
        ]
    
    target_str = str(target)
    if os.name == "nt":
        target_str = target_str.lower()

    for allowed in allowed_dirs:
        allowed_str = str(allowed)
        if os.name == "nt":
            allowed_str = allowed_str.lower()
        if target_str.startswith(allowed_str):
            return target
        try:
            target.relative_to(allowed)
            return target
        except ValueError:
            continue
    raise ValueError(f"Path '{path_str}' is outside the sandbox. Allowed: {[str(d) for d in allowed_dirs]}")


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
    safe = _assert_safe_path(path, write_op=False)
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
    safe = _assert_safe_path(path, write_op=False)
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
    safe = _assert_safe_path(path, write_op=True)
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
    # Cap to prevent unbounded memory growth
    if len(_memory_store) > 1000:
        _memory_store.pop(0)

    log_path = Path("./outputs/memory_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return f"Event logged: [{category}] {content}"


# Global LLM and Config reference for tool usage
_LLM_CLIENT = None
_CONFIG = None


def _fallback_classify_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".go", ".java", ".cpp", ".c", ".h", ".sh", ".ps1", ".bat"}:
        return "code"
    if ext in {".txt", ".md", ".pdf", ".docx", ".doc", ".rtf"}:
        return "documentation"
    if ext in {".csv", ".xlsx", ".xls", ".json", ".xml", ".yaml", ".yml", ".ini"}:
        return "data"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp"}:
        return "images"
    if ext in {".mp3", ".wav", ".mp4", ".mkv", ".avi", ".mov"}:
        return "media"
    if ext in {".log", ".err", ".out"}:
        return "logs"
    if ext in {".zip", ".tar", ".gz", ".rar", ".7z"}:
        return "archives"
    return "others"


async def sort_files(directory: str = "./workspace", output_dir: str = "./workspace") -> str:
    """
    Sorts files in a sandboxed directory into subfolders according to their content using LLM classification.
    """
    safe_dir = _assert_safe_path(directory, write_op=False)
    safe_output = _assert_safe_path(output_dir, write_op=True)

    if not safe_dir.exists():
        return f"Source directory '{directory}' does not exist."
    if not safe_dir.is_dir():
        return f"Source path '{directory}' is not a directory."

    # List all files (excluding directories) sorted alphabetically
    files = sorted([e for e in safe_dir.iterdir() if e.is_file()], key=lambda x: x.name)
    if not files:
        return f"No files found to sort in '{directory}'."

    sorted_count = 0
    results = []

    for file_path in files:
        # Avoid sorting configuration or special system files in project root
        if file_path.name in {".env", "jarvis_env", "jarvis_voice_section.ini", "desktop_automation_report.md"}:
            continue
        
        # Read a snippet of the file content
        snippet = ""
        try:
            size = file_path.stat().st_size
            if size > 0:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    snippet = f.read(2000)
        except Exception as exc:
            logger.warning(f"Could not read content snippet from {file_path.name}: {exc}")

        # Classification prompt
        prompt = (
            "Analyze the following file name and content snippet. Determine the best, most descriptive single-word or short phrase folder name to categorize this file. "
            "Examples of folder names: 'code', 'documentation', 'finance', 'logs', 'images', 'data', 'others'.\n\n"
            f"File Name: {file_path.name}\n"
            f"Snippet:\n{snippet}\n\n"
            "Respond with ONLY the folder name. Do NOT write any thinking process, explanation, quotes, or markdown. Output only the category name."
        )

        category = "others"
        if _LLM_CLIENT is not None:
            try:
                raw_response = await _LLM_CLIENT.complete(
                    prompt,
                    system="You are a file organization assistant. You only output a single clean directory name.",
                    temperature=0.1,
                    task_type="chat"
                )
                # Clean LLM response (remove thinking block, markdown formatting, quotes, etc.)
                import re
                cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
                cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
                # Keep only word characters (alphanumeric and underscore/hyphen)
                cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip()
                # Replace multiple spaces/newlines with a single underscore or hyphen
                cleaned = re.sub(r"\s+", "_", cleaned)
                if cleaned:
                    category = cleaned.lower()
            except Exception as exc:
                logger.warning(f"LLM classification failed for {file_path.name}, using fallback: {exc}")
                category = _fallback_classify_file(file_path)
        else:
            category = _fallback_classify_file(file_path)

        # Ensure category folder exists
        target_folder = safe_output / category
        try:
            target_folder.mkdir(parents=True, exist_ok=True)
            target_file_path = target_folder / file_path.name
            
            # Move the file
            import shutil
            shutil.move(str(file_path), str(target_file_path))
            sorted_count += 1
            results.append(f"{file_path.name} -> {category}/")
        except Exception as exc:
            results.append(f"FAILED to move {file_path.name}: {exc}")

    return f"Successfully sorted {sorted_count}/{len(files)} files.\nDetails:\n" + "\n".join(results)


async def find_files(pattern: str, directory: str = "./workspace") -> str:
    """Finds files matching a wildcard pattern in a sandboxed directory (recursive)."""
    safe_dir = _assert_safe_path(directory, write_op=False)
    if not safe_dir.exists():
        return f"Directory '{directory}' does not exist."
    
    matches = []
    ignored_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "jarvis_env"}
    
    for root, dirs, files in os.walk(safe_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        try:
            _assert_safe_path(root, write_op=False)
        except PermissionError:
            continue
            
        import fnmatch
        for filename in fnmatch.filter(files, pattern):
            file_path = Path(root) / filename
            try:
                rel_path = file_path.relative_to(safe_dir)
                matches.append(f"[FILE] {rel_path} ({file_path.stat().st_size} bytes)")
            except Exception:
                pass
                
        for dirname in fnmatch.filter(dirs, pattern):
            dir_path = Path(root) / dirname
            try:
                rel_path = dir_path.relative_to(safe_dir)
                matches.append(f"[DIR]  {rel_path}")
            except Exception:
                pass
                
    if not matches:
        return f"No matches found for '{pattern}' in '{directory}'."
    return "\n".join(matches)


async def copy_file(source: str, destination: str) -> str:
    """Copies a file from source to destination in the sandbox."""
    safe_src = _assert_safe_path(source, write_op=False)
    safe_dst = _assert_safe_path(destination, write_op=True)
    
    if not safe_src.exists():
        return f"Source file '{source}' does not exist."
    if not safe_src.is_file():
        return f"Source path '{source}' is not a file."
        
    safe_dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(safe_src), str(safe_dst))
    return f"Successfully copied '{source}' to '{destination}'."


async def move_file(source: str, destination: str) -> str:
    """Moves a file or directory from source to destination in the sandbox."""
    safe_src = _assert_safe_path(source, write_op=True)
    safe_dst = _assert_safe_path(destination, write_op=True)
    
    if not safe_src.exists():
        return f"Source '{source}' does not exist."
        
    safe_dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.move(str(safe_src), str(safe_dst))
    return f"Successfully moved '{source}' to '{destination}'."


async def create_directory(path: str) -> str:
    """Creates a new directory (and any parent directories) in the sandbox."""
    safe = _assert_safe_path(path, write_op=True)
    safe.mkdir(parents=True, exist_ok=True)
    return f"Successfully created directory '{path}'."


async def fast_search(path: str = "all", query: str = "", content: str = "", threads: int = 8, case_sensitive: bool = False, no_skip: bool = False, max_results: int = 1000) -> str:
    """
    Search files by name pattern and/or by file content (grep) across the PC/drive.
    Highly optimized multi-threaded execution.
    """
    from core.tools.fast_search_tool import run_fast_search
    try:
        res = await run_fast_search(path, query, content, threads, case_sensitive, no_skip, max_results)
        results = res.get("results", [])
        summary = res.get("summary", {})
        
        output = [
            f"Search completed using {res.get('engine', 'unknown')} engine.",
            f"Elapsed time: {summary.get('elapsed')}",
            f"Files scanned: {summary.get('files_scanned')}",
            f"Folders scanned: {summary.get('dirs_scanned')}",
            f"Matches found: {len(results)}\n",
            "Matches:"
        ]
        for item in results[:100]: # display first 100
            if item["type"] == "file":
                output.append(f"[FILE] {item['path']}")
            else:
                output.append(f"[MATCH] {item['path']}:{item['line']}: {item['text']}")
        if len(results) > 100:
            output.append(f"... and {len(results) - 100} more matches.")
        return "\n".join(output)
    except Exception as e:
        return f"Error executing fast search: {e}"


async def convert_file_format(source_path: str, target_format: str, output_path: str = None) -> str:
    """
    Convert a file from its current format to target_format (e.g. webp, pdf, html, csv, json, xlsx, mp3, wav, mp4).
    Dynamically installs missing libraries on demand.
    """
    from core.tools.universal_converter import perform_conversion
    try:
        dest_path = perform_conversion(source_path, target_format, output_path)
        return f"File successfully converted! Saved to: {dest_path}"
    except Exception as e:
        return f"Error converting file: {e}"


def register_all_tools(router, llm=None, config=None) -> None:
    """Register all built-in tools with a ToolRouter instance."""
    global _LLM_CLIENT, _CONFIG
    _LLM_CLIENT = llm
    _CONFIG = config
    allow_gui_automation = False
    allow_app_launch = True
    if config is not None:
        try:
            allow_gui_automation = config.getboolean(
                "execution",
                "allow_gui_automation",
                fallback=False,
            )
        except Exception:
            allow_gui_automation = False
        try:
            allow_app_launch = config.getboolean(
                "execution",
                "allow_app_launch",
                fallback=True,
            )
        except Exception:
            allow_app_launch = True

    from core.tools.system_automation import (
        async_delete_file,
        async_execute_shell,
        async_launch_application,
        async_write_file,
    )
    # ── Core tools ─────────────────────────────────────────────────────────
    router.register("get_time", get_time)
    router.register("get_system_stats", get_system_stats)
    router.register("list_directory", list_directory)
    router.register("read_file", read_file)
    router.register("write_file", async_write_file)
    router.register("delete_file", async_delete_file)
    router.register("sort_files", sort_files)
    router.register("find_files", find_files)
    router.register("copy_file", copy_file)
    router.register("move_file", move_file)
    router.register("create_directory", create_directory)
    if allow_app_launch:
        router.register("launch_application", async_launch_application)
    router.register("execute_shell", async_execute_shell)
    router.register("write_file_safe", write_file_safe)
    router.register("search_memory", search_memory)
    router.register("log_event", log_event)
    router.register("fast_search", fast_search)
    router.register("convert_file_format", convert_file_format)

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
            describe_screen,
            find_text_on_screen,
            read_screen_text,
            wait_for_text_on_screen,
        )
        from core.tools.gui_control import get_active_window
        router.register("capture_screen", capture_screen)
        router.register("capture_region", capture_region)
        router.register("find_text_on_screen", find_text_on_screen)
        router.register("read_screen_text", read_screen_text)
        router.register("wait_for_text_on_screen", wait_for_text_on_screen)
        router.register("describe_screen", describe_screen)
        router.register("get_active_window", get_active_window)
        logger.info("Screen tools registered (Session 7)")
    except Exception as e:
        logger.warning("Screen tools unavailable: %s", e)

    # ── GUI control tools (Session 7) ──────────────────────────────────────
    if allow_gui_automation:
        try:
            from core.tools.gui_control import (
                click,
                click_screen_target,
                click_text_on_screen,
                clipboard_get,
                clipboard_paste,
                clipboard_set,
                double_click,
                double_click_screen_target,
                drag,
                focus_window,
                move_mouse,
                press_key,
                right_click,
                right_click_screen_target,
                scroll,
                type_text,
                hotkey,
            )
            router.register("click", click)
            router.register("double_click", double_click)
            router.register("right_click", right_click)
            router.register("type_text", type_text)
            router.register("hotkey", hotkey)
            router.register("press_key", press_key)
            router.register("move_mouse", move_mouse)
            router.register("scroll", scroll)
            router.register("drag", drag)
            router.register("focus_window", focus_window)
            router.register("clipboard_get", clipboard_get)
            router.register("clipboard_set", clipboard_set)
            router.register("clipboard_paste", clipboard_paste)
            router.register("click_text_on_screen", click_text_on_screen)
            router.register("click_screen_target", click_screen_target)
            router.register("double_click_screen_target", double_click_screen_target)
            router.register("right_click_screen_target", right_click_screen_target)
            logger.info("GUI control tools registered (Session 7)")
        except Exception as e:
            logger.warning("GUI control tools unavailable: %s", e)
    else:
        logger.info("GUI control tools skipped because allow_gui_automation=false")

    # ── Web Research tools ─────────────────────────────────────────────────
    try:
        from core.tools.web_tools import (
            configure_web_tools,
            web_search,
            web_scrape,
        )
        configure_web_tools(config=config, llm=llm)
        router.register("web_search", web_search)
        router.register("web_scrape", web_scrape)
        logger.info("Web research tools registered")
    except Exception as e:
        logger.warning("Web research tools unavailable: %s", e)

    logger.info("Registered %d tools total: %s", len(router.registered_tools()), router.registered_tools())
