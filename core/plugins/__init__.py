"""Unified plugin catalog stub for Jarvis extensions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("Jarvis.Plugins")


class PluginCatalog:
    """Mock/Stub of PluginCatalog to support the Dashboard summary API after manifest removal."""
    def __init__(self, root: str | Path | None = None, config: Any | None = None, enabled_scopes: set[str] | None = None) -> None:
        self.root = Path(root or "core/plugins")
        self.enabled_scopes = enabled_scopes or set()
        self.errors: dict[str, str] = {}

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "count": 0,
            "enabled_scopes": sorted(list(self.enabled_scopes)),
            "plugins": [],
            "errors": {},
        }


class PluginManifest:
    pass


class PluginManifestError(ValueError):
    pass


def load_plugin_manifest(plugin_dir: Any) -> Any:
    return None


__all__ = [
    "PluginCatalog",
    "PluginManifest",
    "PluginManifestError",
    "load_plugin_manifest",
]
