"""
Import Validator, Safe Import Utility, Dependency Scanner, and Runtime Protection Wrapper.
Prevents unhandled ModuleNotFoundErrors and circular dependency failures from crashing the runtime.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =====================================================================
# 1. Safe Import Utility & Fallback Wrapper
# =====================================================================

class FallbackMock:
    """Mock object that acts as a fallback for missing modules, logging warnings when accessed."""

    def __init__(self, name: str, reason: str = "") -> None:
        self.__name = name
        self.__reason = reason

    def __getattr__(self, item: str) -> Any:
        logger.warning(
            "Accessing attribute %r on missing/mock module %r (Reason: %s)",
            item, self.__name, self.__reason
        )
        return FallbackMock(f"{self.__name}.{item}", self.__reason)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Calling missing/mock module/function %r (Reason: %s)",
            self.__name, self.__reason
        )
        return None


def safe_import(module_name: str, fallback_obj: Any = None) -> Any:
    """
    Attempt to import a module. Returns the imported module,
    or a fallback mock object if the module is missing or fails to load.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logger.warning(
            "Safe import failed for module %r, utilizing fallback/mock. Error: %s",
            module_name, exc
        )
        if fallback_obj is not None:
            return fallback_obj
        return FallbackMock(module_name, str(exc))


# =====================================================================
# 2. Runtime Protection Wrapper / Crash Boundary
# =====================================================================

def protect_runtime(fallback_value: Any = None) -> Callable[[F], F]:
    """
    Decorator to wrap sync or async functions in a runtime safety boundary.
    Catches all exceptions, logs them, and returns a fallback value instead of crashing.
    """
    def decorator(func: F) -> F:
        import asyncio

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    logger.error(
                        "Runtime crash prevented in async function %r: %s",
                        func.__name__, exc, exc_info=True
                    )
                    return fallback_value
            return cast(F, async_wrapper)
        else:
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.error(
                        "Runtime crash prevented in sync function %r: %s",
                        func.__name__, exc, exc_info=True
                    )
                    return fallback_value
            return cast(F, sync_wrapper)

    return decorator


# =====================================================================
# 3. Missing Module Detector & Dependency Scanner (AST Based)
# =====================================================================

@dataclass
class ImportDiagnostic:
    file_path: Path
    line_number: int
    raw_import_string: str
    target_module: str
    is_relative: bool
    status: str  # 'OK', 'FAIL', 'CIRCULAR'
    error_message: str = ""


class DependencyScanner:
    """Scans codebase python files, extracts imports via AST, and validates they resolve."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()

    def scan_project(self) -> list[ImportDiagnostic]:
        root_dir_str = str(self.root_dir)
        if root_dir_str not in sys.path:
            sys.path.insert(0, root_dir_str)
        diagnostics: list[ImportDiagnostic] = []
        for root, _, files in os.walk(self.root_dir):
            # Ignore environment and git directories
            if any(folder in root for folder in ("jarvis_env", ".git", "__pycache__", ".venv", "venv", ".mypy_cache")):
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    diagnostics.extend(self._scan_file(file_path))

        return diagnostics

    def _scan_file(self, file_path: Path) -> list[ImportDiagnostic]:
        file_diagnostics: list[ImportDiagnostic] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content, filename=str(file_path))
        except Exception as exc:
            logger.error("Failed to parse AST for %s: %s", file_path, exc)
            return file_diagnostics

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    diag = self._validate_import(file_path, node.lineno, alias.name, f"import {alias.name}", False)
                    file_diagnostics.append(diag)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    is_relative = node.level > 0
                    diag = self._validate_import(
                        file_path,
                        node.lineno,
                        node.module,
                        f"from {'.' * node.level}{node.module} import ...",
                        is_relative,
                        level=node.level
                    )
                    file_diagnostics.append(diag)

        return file_diagnostics

    def _validate_import(
        self,
        file_path: Path,
        lineno: int,
        module_name: str,
        raw_str: str,
        is_relative: bool,
        level: int = 0
    ) -> ImportDiagnostic:
        # Determine fully qualified module name
        fq_name = module_name
        if is_relative:
            # Resolve relative import to package root path
            rel_package = file_path.parent
            for _ in range(level - 1):
                rel_package = rel_package.parent
            
            # Match package folder structure to import paths
            parts = []
            curr = rel_package
            while curr != self.root_dir and curr != curr.parent:
                parts.append(curr.name)
                curr = curr.parent
            parts.reverse()
            fq_name = ".".join(parts + [module_name]) if parts else module_name

        try:
            # Attempt static resolution or spec checking to avoid executing module at top level
            spec = None
            # Check sys.modules cache
            if fq_name in sys.modules:
                spec = getattr(sys.modules[fq_name], "__spec__", None)
            
            if spec is None:
                spec = importlib.util.find_spec(fq_name)

            if spec is None:
                # Try finding as nested file/package in sys.path
                raise ModuleNotFoundError(f"No spec found for module {fq_name}")

            return ImportDiagnostic(
                file_path=file_path,
                line_number=lineno,
                raw_import_string=raw_str,
                target_module=fq_name,
                is_relative=is_relative,
                status="OK"
            )
        except Exception as exc:
            return ImportDiagnostic(
                file_path=file_path,
                line_number=lineno,
                raw_import_string=raw_str,
                target_module=fq_name,
                is_relative=is_relative,
                status="FAIL",
                error_message=str(exc)
            )


# =====================================================================
# 4. Startup Validator & Import Health Checker
# =====================================================================

class StartupValidator:
    """Verifies that all core controllers, tools, and memory submodules can resolve imports."""

    CRITICAL_MODULES = [
        "core.controller_v2",
        "core.controller.intents",
        "core.controller.intent_router",
        "core.controller.services",
        "core.controller.request_rules",
        "core.controller.web_search",
        "core.tools.builtin_tools",
        "core.tools.web_tools",
        "core.memory.hybrid_memory",
        "core.memory.semantic_memory",
        "core.runtime.paths",
        "core.runtime.bootstrap",
    ]

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def run_preflight_checks(self) -> dict[str, Any]:
        """Perform preflight checks on critical imports and returns a summary health report."""
        root_dir_str = str(self.root_dir)
        if root_dir_str not in sys.path:
            sys.path.insert(0, root_dir_str)
        report = {
            "status": "GREEN",
            "passed": [],
            "failed": [],
        }

        for module in self.CRITICAL_MODULES:
            try:
                importlib.util.find_spec(module)
                # Test loading
                importlib.import_module(module)
                report["passed"].append(module)
            except Exception as exc:
                logger.error("Preflight validation failed for %r: %s", module, exc)
                report["failed"].append({"module": module, "error": str(exc)})
                report["status"] = "RED"

        return report

    def generate_dependency_graph(self) -> dict[str, list[str]]:
        """Static AST scan to map modules to their dependencies."""
        scanner = DependencyScanner(self.root_dir)
        diags = scanner.scan_project()

        graph: dict[str, list[str]] = {}
        for diag in diags:
            # Map source relative path to target imports
            rel_src = str(diag.file_path.relative_to(self.root_dir)).replace(os.sep, ".")
            if rel_src.endswith(".py"):
                rel_src = rel_src[:-3]
            
            if rel_src not in graph:
                graph[rel_src] = []
            
            if diag.target_module not in graph[rel_src]:
                graph[rel_src].append(diag.target_module)

        return graph


def run_diagnostics(root_dir: Path) -> None:
    """Runs a complete diagnostics scan and prints it out."""
    root_dir_str = str(root_dir)
    if root_dir_str not in sys.path:
        sys.path.insert(0, root_dir_str)
    print("====================================================")
    print("Python Runtime Recovery - Preflight Diagnostics Scan")
    print("====================================================")

    validator = StartupValidator(root_dir)
    preflight = validator.run_preflight_checks()

    print(f"\nCritical Modules Preflight: {preflight['status']}")
    print(f"Passed: {len(preflight['passed'])} modules")
    if preflight["failed"]:
        print(f"Failed: {len(preflight['failed'])} modules")
        for fail in preflight["failed"]:
            print(f"  - {fail['module']}: {fail['error']}")

    print("\nAST Dependency Scanner Report:")
    scanner = DependencyScanner(root_dir)
    diags = scanner.scan_project()
    failed_imports = [d for d in diags if d.status == "FAIL"]

    if not failed_imports:
        print("  [OK] No missing dependencies detected via static analysis.")
    else:
        print(f"  [WARN] Found {len(failed_imports)} broken imports in project files:")
        for diag in failed_imports:
            rel_path = diag.file_path.relative_to(root_dir)
            print(f"    - {rel_path}:{diag.line_number} -> Failed to resolve: {diag.target_module} ({diag.error_message})")

    print("====================================================")


if __name__ == "__main__":
    # Resolve project root relative to this file
    proj_root = Path(__file__).resolve().parents[2]
    run_diagnostics(proj_root)
