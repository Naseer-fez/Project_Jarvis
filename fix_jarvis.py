#!/usr/bin/env python3
"""
fix_jarvis.py — Idempotent patch script for Project_Jarvis-main.
Run from the project root: python fix_jarvis.py
"""

import os
import sys
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"archive_jarvis_duplicate", "archive_legacy", "Failed", "__pycache__",
             ".git", ".github", "jarvis_env", "venv", ".env", ".venv"}

CHANGED = []
ERRORS = []


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def info(msg):
    print(f"  [FIX] {msg}")
    CHANGED.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}")


def die(msg):
    print(f"\n  [FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_init(pkg_dir: Path):
    init = pkg_dir / "__init__.py"
    if not init.exists():
        write(init, "")
        info(f"Created missing __init__.py: {init.relative_to(ROOT)}")


def patch_file(path: Path, old: str, new: str, description: str) -> bool:
    content = read(path)
    if old in content:
        write(path, content.replace(old, new))
        info(f"{description} — {path.relative_to(ROOT)}")
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Step 1: Verify we are in the right directory
# ─────────────────────────────────────────────────────────────

print("\n=== Step 1: Verifying project root ===")
if not (ROOT / "main.py").exists():
    die(f"main.py not found in {ROOT}. Run this script from the project root directory.")
if not (ROOT / "core").exists():
    die(f"core/ not found in {ROOT}. Run this script from the project root directory.")
print(f"  Project root: {ROOT}")


# ─────────────────────────────────────────────────────────────
# Step 2: Ensure required __init__.py files exist
# ─────────────────────────────────────────────────────────────

print("\n=== Step 2: Ensuring __init__.py files exist ===")

required_packages = [
    ROOT / "core",
    ROOT / "core" / "agent",
    ROOT / "core" / "agentic",
    ROOT / "core" / "autonomy",
    ROOT / "core" / "controller",
    ROOT / "core" / "execution",
    ROOT / "core" / "hardware",
    ROOT / "core" / "introspection",
    ROOT / "core" / "llm",
    ROOT / "core" / "logging",
    ROOT / "core" / "memory",
    ROOT / "core" / "metrics",
    ROOT / "core" / "planning",
    ROOT / "core" / "tools",
    ROOT / "core" / "vision",
    ROOT / "core" / "voice",
    ROOT / "integrations",
    ROOT / "integrations" / "clients",
    ROOT / "integrations" / "tests",
    ROOT / "memory",
    ROOT / "tests",
]

for pkg in required_packages:
    if pkg.exists():
        ensure_init(pkg)

# Create audit/ package (imported by core/agent/controller.py as "audit.audit_logger")
audit_dir = ROOT / "audit"
audit_dir.mkdir(exist_ok=True)
ensure_init(audit_dir)

# Copy audit_logger.py into audit/ package if not already there
audit_logger_src = ROOT / "core" / "logging" / "audit_logger.py"
audit_logger_dst = ROOT / "audit" / "audit_logger.py"
if not audit_logger_dst.exists():
    if audit_logger_src.exists():
        shutil.copy2(audit_logger_src, audit_logger_dst)
        info(f"Copied audit_logger.py -> audit/audit_logger.py")
    else:
        # Create a stub audit_logger.py
        stub = textwrap.dedent('''\
            """
            audit/audit_logger.py — stub shim for core/agent/controller.py compatibility.
            Delegates to core.logging.audit_logger.AuditLogger if available,
            otherwise provides a minimal no-op implementation.
            """
            import json
            import logging
            import time
            from pathlib import Path
            from typing import Any

            logger = logging.getLogger("Jarvis.AuditLogger")
            OUTPUTS_DIR = Path("./outputs/Jarvis-Session/")


            class AuditLogger:
                def __init__(self, session_id: str):
                    self.session_id = session_id
                    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                    self._log_path = OUTPUTS_DIR / f"audit_{session_id}.jsonl"

                def log(self, event_type: str, data: Any = None) -> None:
                    entry = {
                        "ts": time.time(),
                        "session_id": self.session_id,
                        "event": event_type,
                        "data": data,
                    }
                    try:
                        with self._log_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(entry, default=str) + "\\n")
                    except OSError as exc:
                        logger.warning("AuditLogger write failed: %s", exc)

                def log_plan(self, plan: Any) -> None:
                    self.log("plan", data=plan)

                def log_observation(self, obs: Any) -> None:
                    self.log("observation", data=obs)

                def log_risk(self, risk: Any) -> None:
                    self.log("risk", data=risk)

                def log_reflection(self, reflection: str) -> None:
                    self.log("reflection", data={"reflection": reflection})

                def log_final(self, response: str, success: bool) -> None:
                    self.log("final", data={"response": response, "success": success})
            ''')
        write(audit_logger_dst, stub)
        info("Created stub audit/audit_logger.py")


# ─────────────────────────────────────────────────────────────
# Step 3: Create core/planning/task_planner.py shim
# ─────────────────────────────────────────────────────────────

print("\n=== Step 3: Creating missing module shims ===")

planning_task_planner = ROOT / "core" / "planning" / "task_planner.py"
if not planning_task_planner.exists():
    shim = textwrap.dedent('''\
        """
        core/planning/task_planner.py
        Shim that re-exports TaskPlanner from core.llm.task_planner for
        backward compatibility with core.agent.controller imports.
        """
        from core.llm.task_planner import TaskPlanner

        __all__ = ["TaskPlanner"]
        ''')
    write(planning_task_planner, shim)
    info("Created core/planning/task_planner.py shim")

# Create core/planning/plan_schema.py shim (core.llm.task_planner imports this)
planning_plan_schema = ROOT / "core" / "planning" / "plan_schema.py"
if not planning_plan_schema.exists():
    real_plan_schema = ROOT / "core" / "llm" / "plan_schema.py"
    if real_plan_schema.exists():
        shim = textwrap.dedent('''\
            """
            core/planning/plan_schema.py
            Shim that re-exports from core.llm.plan_schema for backward compatibility.
            """
            from core.llm.plan_schema import build_unknown_plan, normalize_plan

            __all__ = ["build_unknown_plan", "normalize_plan"]
            ''')
        write(planning_plan_schema, shim)
        info("Created core/planning/plan_schema.py shim")


# Create core/agent/state_machine.py shim (agent_loop and controller import from here)
agent_state_machine = ROOT / "core" / "agent" / "state_machine.py"
if not agent_state_machine.exists():
    real_state_machine = ROOT / "core" / "controller" / "state_machine.py"
    if real_state_machine.exists():
        shim = textwrap.dedent('''\
            """
            core/agent/state_machine.py
            Shim that re-exports StateMachine and AgentState from
            core.controller.state_machine for import compatibility.
            """
            from core.controller.state_machine import StateMachine, AgentState

            __all__ = ["StateMachine", "AgentState"]
            ''')
        write(agent_state_machine, shim)
        info("Created core/agent/state_machine.py shim")
    else:
        warn("core/controller/state_machine.py not found — cannot create shim")


# ─────────────────────────────────────────────────────────────
# Step 4: Fix core/agent/agent_loop.py broken imports
# ─────────────────────────────────────────────────────────────

print("\n=== Step 4: Fixing core/agent/agent_loop.py imports ===")

agent_loop = ROOT / "core" / "agent" / "agent_loop.py"
if agent_loop.exists():
    content = read(agent_loop)
    original = content

    # Fix: core.state_machine -> core.agent.state_machine
    content = content.replace(
        "from core.state_machine import StateMachine, AgentState",
        "from core.agent.state_machine import StateMachine, AgentState"
    )

    # Fix: core.task_planner -> core.llm.task_planner
    content = content.replace(
        "from core.task_planner import TaskPlanner, Plan, PlanStep",
        "from core.llm.task_planner import TaskPlanner"
    )

    # Fix: core.tool_router -> core.tools.tool_router
    content = content.replace(
        "from core.tool_router import ToolRouter, ToolObservation",
        "from core.tools.tool_router import ToolRouter, ToolObservation"
    )

    # Fix: core.risk_evaluator -> core.autonomy.risk_evaluator
    content = content.replace(
        "from core.risk_evaluator import RiskEvaluator, RiskLevel",
        "from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel"
    )

    # Fix: core.autonomy_governor -> core.autonomy.autonomy_governor
    content = content.replace(
        "from core.autonomy_governor import AutonomyGovernor",
        "from core.autonomy.autonomy_governor import AutonomyGovernor"
    )

    if content != original:
        write(agent_loop, content)
        info("Fixed imports in core/agent/agent_loop.py")

    # Ensure Plan and PlanStep stubs exist if referenced in the file body
    # (they were imported but may not exist; add a local shim after imports)
    content_after = read(agent_loop)
    if "Plan" in content_after and "class Plan" not in content_after and "PlanStep" in content_after:
        # Insert stub dataclasses after the imports section if not already present
        stub_insertion = textwrap.dedent('''\

            # --- Compatibility stubs (Plan/PlanStep not exported by task_planner) ---
            try:
                from core.llm.task_planner import Plan, PlanStep  # type: ignore[attr-defined]
            except ImportError:
                from dataclasses import dataclass, field as _field
                from typing import List as _List

                @dataclass
                class PlanStep:
                    tool: str = ""
                    args: dict = _field(default_factory=dict)
                    description: str = ""

                @dataclass
                class Plan:
                    goal: str = ""
                    steps: _List[PlanStep] = _field(default_factory=list)
            # --- End stubs ---
            ''')
        # Insert after the last import block
        lines = content_after.splitlines(keepends=True)
        insert_after = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_after = i
        marker = "# --- Compatibility stubs"
        if marker not in content_after:
            lines.insert(insert_after + 1, stub_insertion)
            write(agent_loop, "".join(lines))
            info("Added Plan/PlanStep compat stubs in core/agent/agent_loop.py")
else:
    warn("core/agent/agent_loop.py not found — skipping")


# ─────────────────────────────────────────────────────────────
# Step 5: Fix core/agent/controller.py imports
# ─────────────────────────────────────────────────────────────

print("\n=== Step 5: Fixing core/agent/controller.py imports ===")

agent_controller = ROOT / "core" / "agent" / "controller.py"
if agent_controller.exists():
    content = read(agent_controller)
    original = content

    # Fix: audit.audit_logger is now in our new audit/ package — no change needed
    # but let's make it robust with a try/except in the file
    bare_import = "from audit.audit_logger import AuditLogger"
    guarded_import = textwrap.dedent("""\
        try:
            from audit.audit_logger import AuditLogger
        except ImportError:
            from core.logging.audit_logger import AuditLogger""")
    if bare_import in content and guarded_import not in content:
        content = content.replace(
            bare_import,
            guarded_import
        )

    if content != original:
        write(agent_controller, content)
        info("Fixed audit import in core/agent/controller.py")
else:
    warn("core/agent/controller.py not found — skipping")


# ─────────────────────────────────────────────────────────────
# Step 6: Fix main.py — Controller class name mismatch
# ─────────────────────────────────────────────────────────────

print("\n=== Step 6: Fixing Controller alias in core/controller_v2.py ===")

ctrl_v2 = ROOT / "core" / "controller_v2.py"
if ctrl_v2.exists():
    content = read(ctrl_v2)
    # main.py imports "Controller" from core.controller_v2 but the class is JarvisControllerV2
    # Add an alias at the bottom of the file if not already present
    if "class JarvisControllerV2" in content and "Controller = JarvisControllerV2" not in content:
        content += "\n\n# Alias for backward-compatible import in main.py\nController = JarvisControllerV2\n"
        write(ctrl_v2, content)
        info("Added Controller alias to core/controller_v2.py")

    # main.py calls controller.start(), controller.run_cli(), controller.shutdown()
    # Make sure these methods exist on JarvisControllerV2
    missing_methods = []
    if "async def start(" not in content and "def start(" not in content:
        missing_methods.append("start")
    if "async def run_cli(" not in content and "def run_cli(" not in content:
        missing_methods.append("run_cli")
    if "async def shutdown(" not in content and "def shutdown(" not in content:
        missing_methods.append("shutdown")

    if missing_methods:
        stubs = "\n"
        for m in missing_methods:
            if m == "start":
                stubs += textwrap.dedent("""\
                    async def start(self):
                        \"\"\"Initialize and start the controller.\"\"\"
                        self.initialize()

                    """)
            elif m == "run_cli":
                stubs += textwrap.dedent("""\
                    async def run_cli(self):
                        \"\"\"Interactive CLI loop.\"\"\"
                        import asyncio
                        print(f"Jarvis V2 ready (session {self.session_id}). Type 'exit' to quit.")
                        loop = asyncio.get_event_loop()
                        while True:
                            try:
                                user_input = await loop.run_in_executor(None, input, "You: ")
                            except EOFError:
                                break
                            if user_input.strip().lower() in ("exit", "quit"):
                                break
                            response = self.process(user_input)
                            print(f"Jarvis: {response}")

                    """)
            elif m == "shutdown":
                stubs += textwrap.dedent("""\
                    async def shutdown(self):
                        \"\"\"Graceful shutdown.\"\"\"
                        pass

                    """)

        # Insert stubs before the Controller alias line
        alias_marker = "# Alias for backward-compatible"
        if alias_marker in content:
            content = content.replace(alias_marker, stubs + alias_marker)
        else:
            content += stubs
        write(ctrl_v2, content)
        info(f"Added missing async methods to JarvisControllerV2: {', '.join(missing_methods)}")
else:
    warn("core/controller_v2.py not found — skipping")


# ─────────────────────────────────────────────────────────────
# Step 7: Fix core/logging/ shadowing stdlib logging module
# ─────────────────────────────────────────────────────────────

print("\n=== Step 7: Fixing core/logging/ stdlib shadow conflict ===")

logging_init = ROOT / "core" / "logging" / "__init__.py"
if logging_init.exists():
    content = read(logging_init)
    # The __init__.py imports from .logger which imports core.logger fine.
    # The real risk: any file inside core/logging/ that does "import logging"
    # will import the package itself (circular). We fix by using importlib.
    # Scan all .py files inside core/logging/ for bare "import logging"
    logging_dir = ROOT / "core" / "logging"
    for py_file in logging_dir.glob("*.py"):
        fcontent = read(py_file)
        if "import logging\n" in fcontent and "import importlib" not in fcontent:
            # Replace bare stdlib import with a safe form
            safe_import = textwrap.dedent("""\
                import importlib as _importlib
                import sys as _sys
                _stdlib_logging = _importlib.import_module("logging")
                """)
            # We only need to add this if the file actually USES the logging module
            # Check if logging.getLogger or logging.DEBUG etc is used
            if "logging.getLogger" in fcontent or "logging.DEBUG" in fcontent or \
               "logging.INFO" in fcontent or "logging.basicConfig" in fcontent or \
               "logging.StreamHandler" in fcontent or "logging.FileHandler" in fcontent:
                if "import logging\n" in fcontent and "_stdlib_logging" not in fcontent:
                    new_content = fcontent.replace(
                        "import logging\n",
                        safe_import + "\nlogging = _stdlib_logging\n"
                    )
                    write(py_file, new_content)
                    info(f"Fixed stdlib logging shadow in {py_file.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────
# Step 8: Fix core/execution/dispatcher.py — verify imports exist
# ─────────────────────────────────────────────────────────────

print("\n=== Step 8: Checking core/execution/dispatcher.py ===")

dispatcher = ROOT / "core" / "execution" / "dispatcher.py"
if dispatcher.exists():
    content = read(dispatcher)
    original = content
    # autonomy_policy is in core/agentic/ — that import looks correct already
    # system_automation is in core/tools/ — that import looks correct already
    # Nothing to fix here unless the exports are wrong
    system_auto = ROOT / "core" / "tools" / "system_automation.py"
    if system_auto.exists():
        sa_content = read(system_auto)
        # Verify TOOL_REGISTRY, ToolResult, and the async functions exist
        needed = ["TOOL_REGISTRY", "ToolResult", "async_list_directory",
                  "async_read_file", "async_write_file", "async_delete_file",
                  "async_launch_application", "async_execute_shell"]
        missing = [n for n in needed if n not in sa_content]
        if missing:
            warn(f"core/tools/system_automation.py may be missing: {missing}")
        else:
            print(f"  [OK] core/tools/system_automation.py exports look correct")

    if content != original:
        write(dispatcher, content)


# ─────────────────────────────────────────────────────────────
# Step 9: Ensure core/logging/__init__.py re-exports verify_audit
# ─────────────────────────────────────────────────────────────

print("\n=== Step 9: Verifying core/logging/__init__.py exports ===")

if logging_init.exists():
    content = read(logging_init)
    needed_exports = ["setup", "get", "get_logger", "audit", "verify_audit", "AuditLog"]
    missing_exports = [e for e in needed_exports if e not in content]
    if missing_exports:
        # Rewrite the __init__.py to be complete
        new_init = textwrap.dedent("""\
            from .logger import AuditLog, audit, get, get_logger, setup, verify_audit

            __all__ = [
                "AuditLog",
                "setup",
                "get",
                "get_logger",
                "audit",
                "verify_audit",
            ]
            """)
        write(logging_init, new_init)
        info(f"Fixed core/logging/__init__.py missing exports: {missing_exports}")
    else:
        print("  [OK] core/logging/__init__.py exports are complete")


# ─────────────────────────────────────────────────────────────
# Step 10: Fix config path references
# ─────────────────────────────────────────────────────────────

print("\n=== Step 10: Verifying config files exist ===")

config_dir = ROOT / "config"
config_dir.mkdir(exist_ok=True)

# main.py defaults to config/jarvis.ini
jarvis_ini = config_dir / "jarvis.ini"
if not jarvis_ini.exists():
    # Copy from jarvis_config.ini if it exists
    jarvis_config_ini = config_dir / "jarvis_config.ini"
    if jarvis_config_ini.exists():
        shutil.copy2(jarvis_config_ini, jarvis_ini)
        info("Copied config/jarvis_config.ini -> config/jarvis.ini")
    else:
        default_ini = textwrap.dedent("""\
            [general]
            name = Jarvis
            session_name = default

            [logging]
            level = INFO
            app_file = logs/jarvis.log
            audit_file = logs/audit.jsonl

            [voice]
            enabled = false

            [llm]
            model = deepseek-r1:8b
            host = http://localhost:11434

            [memory]
            db_path = memory/memory.db
            chroma_path = data/chroma
            """)
        write(jarvis_ini, default_ini)
        info("Created default config/jarvis.ini")
else:
    print("  [OK] config/jarvis.ini exists")


# ─────────────────────────────────────────────────────────────
# Step 11: Create logs/ and memory/ dirs to prevent runtime errors
# ─────────────────────────────────────────────────────────────

print("\n=== Step 11: Creating required runtime directories ===")

for d in ["logs", "memory", "data", "outputs"]:
    p = ROOT / d
    p.mkdir(exist_ok=True)
    gitkeep = p / ".gitkeep"
    if not any(p.iterdir()) or not gitkeep.exists():
        gitkeep.touch()
print("  [OK] Runtime directories ensured: logs/, memory/, data/, outputs/")


# ─────────────────────────────────────────────────────────────
# Step 12: Remove circular import in core/logging/logger.py
# ─────────────────────────────────────────────────────────────

print("\n=== Step 12: Checking for circular imports ===")

# core/logging/logger.py imports "import core.logger as _core_logger"
# This is fine as long as core/logging is not imported before core.logger
# The real issue: core/__init__.py could import core.logging which imports core.logger
# Check core/__init__.py
core_init = ROOT / "core" / "__init__.py"
if core_init.exists():
    content = read(core_init)
    if "from core.logging" in content or "import core.logging" in content:
        warn("core/__init__.py imports from core.logging — potential circular import risk")
        # Guard it
        new_content = content.replace(
            "from core.logging",
            "# Deferred to avoid circular import — use core.logger directly\n# from core.logging"
        ).replace(
            "import core.logging",
            "# import core.logging  # deferred"
        )
        if new_content != content:
            write(core_init, new_content)
            info("Guarded potential circular import in core/__init__.py")
    else:
        print("  [OK] No circular import detected in core/__init__.py")


# ─────────────────────────────────────────────────────────────
# Step 13a: Strip BOM (U+FEFF) from Python files that have it
# ─────────────────────────────────────────────────────────────

print("\n=== Step 13a: Stripping BOM characters from Python files ===")

BOM = "\ufeff"
bom_fixed = 0
for py_file in ROOT.rglob("*.py"):
    # Skip archived/ignored directories
    parts = py_file.parts
    if any(skip in parts for skip in SKIP_DIRS):
        continue
    try:
        raw = py_file.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            py_file.write_bytes(raw[3:])
            info(f"Stripped BOM from {py_file.relative_to(ROOT)}")
            bom_fixed += 1
    except OSError:
        pass

if bom_fixed == 0:
    print("  [OK] No BOM characters found")


# ─────────────────────────────────────────────────────────────
# Step 13: Scan and report remaining broken imports
# ─────────────────────────────────────────────────────────────

print("\n=== Step 13: Scanning for remaining import issues ===")

import ast

def get_imports(filepath: Path):
    try:
        tree = ast.parse(read(filepath))
    except SyntaxError as e:
        return [], str(e)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
    return imports, None


def module_to_path(module: str) -> Path | None:
    parts = module.split(".")
    # Try as package
    pkg = ROOT.joinpath(*parts) / "__init__.py"
    if pkg.exists():
        return pkg.parent
    # Try as module
    mod = ROOT.joinpath(*parts[:-1]) / (parts[-1] + ".py")
    if mod.exists():
        return mod
    return None


scan_dirs = [
    ROOT / "core" / "agent",
    ROOT / "core" / "agentic",
    ROOT / "core" / "autonomy",
    ROOT / "core" / "execution",
    ROOT / "core" / "memory",
    ROOT / "core" / "planning",
    ROOT / "core" / "tools",
    ROOT / "audit",
]

broken = []
for d in scan_dirs:
    if not d.exists():
        continue
    for py in d.glob("*.py"):
        if py.name.startswith("_") and py.name != "__init__.py":
            continue
        imports, syntax_err = get_imports(py)
        if syntax_err:
            warn(f"Syntax error in {py.relative_to(ROOT)}: {syntax_err}")
            continue
        for imp in imports:
            if imp.startswith("core.") or imp == "core":
                if module_to_path(imp) is None:
                    broken.append((py.relative_to(ROOT), imp))

if broken:
    print("  Remaining unresolved internal imports (may need manual fix):")
    seen = set()
    for filepath, imp in broken:
        key = f"{filepath}:{imp}"
        if key not in seen:
            seen.add(key)
            print(f"    {filepath} -> {imp}")
else:
    print("  [OK] No unresolved internal imports detected in scanned directories")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"DONE — {len(CHANGED)} fix(es) applied:")
for c in CHANGED:
    print(f"  ✓ {c}")

if not CHANGED:
    print("  (no changes needed — already up to date)")

print("\nTo verify the project starts correctly, run:")
print("  python main.py --help")
print("  python -c \"from core.controller_v2 import Controller; print('OK')\"")
print("=" * 60 + "\n")
