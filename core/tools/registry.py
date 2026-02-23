"""
Dynamic tool registry with scoped plugin permissions.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    scope: str
    handler: Callable
    source: str


class ToolRegistry:
    def __init__(self, enabled_scopes: set[str] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._enabled_scopes = enabled_scopes or {"core"}

    def register(self, name: str, handler: Callable, *, scope: str = "core", source: str = "builtin") -> None:
        key = name.strip().lower()
        self._tools[key] = ToolSpec(name=key, scope=scope, handler=handler, source=source)

    def is_permitted(self, name: str) -> bool:
        key = name.strip().lower()
        spec = self._tools.get(key)
        if spec is None:
            return False
        return spec.scope in self._enabled_scopes

    def get_permitted_tools(self) -> dict[str, Callable]:
        out: dict[str, Callable] = {}
        for name, spec in self._tools.items():
            if spec.scope in self._enabled_scopes:
                out[name] = spec.handler
        return out

    def load_plugins(self, plugin_dir: str | Path) -> list[str]:
        directory = Path(plugin_dir)
        if not directory.exists() or not directory.is_dir():
            return []

        loaded: list[str] = []
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"jarvis_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register_fn = getattr(module, "register", None)
            if callable(register_fn):
                register_fn(self)
                loaded.append(path.stem)
        return loaded

    def list_specs(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools.keys())]
