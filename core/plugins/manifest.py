"""Manifest-based plugin loading and validation.

This catalog is intentionally metadata-first. It lets Jarvis discover plugin
capabilities, permissions, UI contributions, workflow nodes, and marketplace
entries without importing plugin code or tying core services to a concrete
implementation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any

logger = logging.getLogger("Jarvis.Plugins")


class PluginManifestError(ValueError):
    """Raised when a plugin manifest is malformed."""


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_KNOWN_TYPES = {
    "workflow_node",
    "ai_tool",
    "ui_widget",
    "theme",
    "data_connector",
    "automation_pack",
    "integration",
    "agent",
}
_KNOWN_PERMISSION_ACCESS = {"read", "write", "execute", "network", "system", "secret"}


@dataclass(frozen=True)
class PluginPermission:
    scope: str
    access: str
    reason: str
    optional: bool = False


@dataclass(frozen=True)
class PluginContribution:
    type: str
    id: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginManifest:
    plugin_id: str
    name: str
    version: str
    api_version: str
    description: str
    author: str
    types: list[str]
    permissions: list[PluginPermission]
    contributions: list[PluginContribution]
    entrypoints: dict[str, str]
    marketplace: dict[str, Any]
    directory: Path
    raw: dict[str, Any] = field(repr=False, compare=False)

    @property
    def is_local_first(self) -> bool:
        return bool(self.marketplace.get("local_first", True))

    @property
    def requires_network(self) -> bool:
        return any(permission.access == "network" for permission in self.permissions)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


class PluginCatalog:
    """Scans plugin directories for manifest.json files."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        config: Any | None = None,
        enabled_scopes: set[str] | None = None,
    ) -> None:
        self.root = _resolve_plugin_root(root=root, config=config)
        self.enabled_scopes = enabled_scopes or _enabled_scopes_from_config(config)
        self._plugins: dict[str, PluginManifest] = {}
        self.errors: dict[str, str] = {}

    def refresh(self) -> list[PluginManifest]:
        self._plugins.clear()
        self.errors.clear()
        if not self.root.exists() or not self.root.is_dir():
            return []

        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            try:
                manifest = load_plugin_manifest(manifest_path.parent)
            except PluginManifestError as exc:
                self.errors[str(manifest_path)] = str(exc)
                continue
            if self._is_enabled(manifest):
                self._plugins[manifest.plugin_id] = manifest
        return self.list_plugins()

    def list_plugins(self) -> list[PluginManifest]:
        if not self._plugins and not self.errors:
            self.refresh()
        return [self._plugins[key] for key in sorted(self._plugins)]

    def get(self, plugin_id: str) -> PluginManifest | None:
        if not self._plugins and not self.errors:
            self.refresh()
        return self._plugins.get(plugin_id)

    def contributions(self, contribution_type: str | None = None) -> list[PluginContribution]:
        items: list[PluginContribution] = []
        for plugin in self.list_plugins():
            for contribution in plugin.contributions:
                if contribution_type is None or contribution.type == contribution_type:
                    items.append(contribution)
        return items

    def summary(self) -> dict[str, Any]:
        plugins = self.list_plugins()
        return {
            "root": str(self.root),
            "count": len(plugins),
            "enabled_scopes": sorted(self.enabled_scopes),
            "plugins": [
                {
                    "id": plugin.plugin_id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "types": plugin.types,
                    "permissions": [permission.scope for permission in plugin.permissions],
                    "contributions": [contribution.id for contribution in plugin.contributions],
                    "local_first": plugin.is_local_first,
                    "requires_network": plugin.requires_network,
                }
                for plugin in plugins
            ],
            "errors": dict(self.errors),
        }

    def _is_enabled(self, manifest: PluginManifest) -> bool:
        if not self.enabled_scopes:
            return True
        return any(permission.scope in self.enabled_scopes for permission in manifest.permissions)


def load_plugin_manifest(plugin_dir: str | Path) -> PluginManifest:
    directory = Path(plugin_dir)
    manifest_path = directory / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise PluginManifestError(f"Plugin manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise PluginManifestError(f"Plugin manifest is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise PluginManifestError(f"Plugin manifest could not be read: {exc}") from exc

    if not isinstance(payload, dict):
        raise PluginManifestError("Plugin manifest root must be a JSON object.")
    return _parse_manifest(payload, directory=directory)


def _parse_manifest(payload: dict[str, Any], *, directory: Path) -> PluginManifest:
    plugin_id = _required_str(payload, "id")
    if not _ID_RE.match(plugin_id):
        raise PluginManifestError("Plugin id must be lowercase and URL-safe.")

    version = _required_str(payload, "version")
    if not _SEMVER_RE.match(version):
        raise PluginManifestError("Plugin version must use semantic versioning, for example 1.0.0.")

    api_version = _required_str(payload, "api_version")
    types = _string_list(payload.get("types", []), "types")
    unknown_types = [plugin_type for plugin_type in types if plugin_type not in _KNOWN_TYPES]
    if unknown_types:
        raise PluginManifestError(f"Unknown plugin type(s): {', '.join(unknown_types)}")

    permissions = [_parse_permission(item) for item in _required_list(payload, "permissions")]
    contributions = [_parse_contribution(item) for item in payload.get("contributions", [])]
    entrypoints = _relative_entrypoints(payload.get("entrypoints", {}))
    marketplace = payload.get("marketplace", {})
    if marketplace is None:
        marketplace = {}
    if not isinstance(marketplace, dict):
        raise PluginManifestError("Plugin marketplace metadata must be an object.")

    return PluginManifest(
        plugin_id=plugin_id,
        name=_required_str(payload, "name"),
        version=version,
        api_version=api_version,
        description=str(payload.get("description", "")).strip(),
        author=str(payload.get("author", "")).strip(),
        types=types,
        permissions=permissions,
        contributions=contributions,
        entrypoints=entrypoints,
        marketplace=marketplace,
        directory=directory,
        raw=payload,
    )


def _parse_permission(item: Any) -> PluginPermission:
    if not isinstance(item, dict):
        raise PluginManifestError("Plugin permissions must be objects.")
    scope = _required_str(item, "scope")
    access = str(item.get("access", "read")).strip().lower()
    if access not in _KNOWN_PERMISSION_ACCESS:
        raise PluginManifestError(f"Unknown plugin permission access: {access}")
    reason = _required_str(item, "reason")
    return PluginPermission(
        scope=scope,
        access=access,
        reason=reason,
        optional=bool(item.get("optional", False)),
    )


def _parse_contribution(item: Any) -> PluginContribution:
    if not isinstance(item, dict):
        raise PluginManifestError("Plugin contributions must be objects.")
    contribution_type = _required_str(item, "type")
    if contribution_type not in _KNOWN_TYPES:
        raise PluginManifestError(f"Unknown plugin contribution type: {contribution_type}")
    contribution_id = _required_str(item, "id")
    name = _required_str(item, "name")
    metadata = item.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise PluginManifestError("Plugin contribution metadata must be an object.")
    return PluginContribution(
        type=contribution_type,
        id=contribution_id,
        name=name,
        metadata=metadata,
    )


def _relative_entrypoints(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PluginManifestError("Plugin entrypoints must be an object.")

    out: dict[str, str] = {}
    for key, raw_path in value.items():
        clean_key = str(key).strip()
        clean_path = str(raw_path).strip()
        if not clean_key or not clean_path:
            continue
        if Path(clean_path).is_absolute() or "://" in clean_path:
            raise PluginManifestError("Plugin entrypoints must be relative paths inside the plugin directory.")
        out[clean_key] = clean_path
    return out


def _resolve_plugin_root(root: str | Path | None, config: Any | None) -> Path:
    if root is None:
        env_root = os.environ.get("JARVIS_PLUGIN_MANIFEST_DIR", "").strip()
        if env_root:
            root = env_root
        elif config is not None:
            getter = getattr(config, "get_str", None)
            if callable(getter):
                root = getter("plugins", "manifest_directory", fallback="plugins")
            else:
                root = config.get("plugins", "manifest_directory", fallback="plugins")
        else:
            root = "plugins"

    candidate = Path(root)
    if candidate.is_absolute():
        return candidate

    from core.runtime.bootstrap import _resolve_path

    return _resolve_path(candidate)


def _enabled_scopes_from_config(config: Any | None) -> set[str]:
    if config is None:
        logger.debug("No configuration provided for plugins; defaulting enabled scopes to {'core'}")
        return {"core"}
    try:
        getter = getattr(config, "get_str", None)
        if callable(getter):
            raw = getter("plugins", "enabled_scopes", fallback="core")
        else:
            raw = config.get("plugins", "enabled_scopes", fallback="core")
    except Exception:
        raw = "core"
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise PluginManifestError(f"Plugin manifest field '{key}' must be a list.")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise PluginManifestError(f"Plugin manifest field '{key}' must be a non-empty string.")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise PluginManifestError(f"Plugin manifest field '{field_name}' must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


__all__ = [
    "PluginCatalog",
    "PluginContribution",
    "PluginManifest",
    "PluginManifestError",
    "PluginPermission",
    "load_plugin_manifest",
]
