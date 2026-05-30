"""Config-backed system blueprint for Jarvis as a local-first AI OS.

The blueprint intentionally stores product and architecture capabilities in a
data file instead of code. Runtime services can inspect it to power onboarding,
dashboard views, marketplace filtering, and future setup wizards without baking
environment-specific assumptions into Python modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class AIOSError(ValueError):
    """Raised when an AI OS blueprint cannot be loaded or validated."""


@dataclass(frozen=True)
class DesignPrinciple:
    id: str
    title: str
    summary: str
    enforcement: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SystemLayer:
    id: str
    name: str
    responsibilities: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    extensibility: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModeDefinition:
    id: str
    name: str
    audience: str
    features: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SystemBlueprint:
    schema_version: str
    product_name: str
    mission: str
    principles: list[DesignPrinciple]
    layers: list[SystemLayer]
    modes: list[ModeDefinition]
    node_categories: list[str]
    provider_types: list[str]
    storage_backends: list[str]
    security_controls: list[str]
    observability: list[str]
    future_extensions: list[str]
    raw: dict[str, Any] = field(repr=False, compare=False)

    @property
    def layer_ids(self) -> set[str]:
        return {layer.id for layer in self.layers}

    @property
    def principle_ids(self) -> set[str]:
        return {principle.id for principle in self.principles}

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def summary(self) -> dict[str, Any]:
        return {
            "product_name": self.product_name,
            "mission": self.mission,
            "principles": [principle.title for principle in self.principles],
            "layers": [layer.name for layer in self.layers],
            "modes": [mode.name for mode in self.modes],
            "node_categories": list(self.node_categories),
            "provider_types": list(self.provider_types),
            "storage_backends": list(self.storage_backends),
            "security_controls": list(self.security_controls),
            "observability": list(self.observability),
            "future_extensions": list(self.future_extensions),
        }


def load_blueprint(path: str | Path | None = None, config: Any | None = None) -> SystemBlueprint:
    """Load the AI OS blueprint from env, config, or the project default."""

    blueprint_path = _resolve_blueprint_path(path=path, config=config)
    try:
        with blueprint_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise AIOSError(f"AI OS blueprint not found: {blueprint_path}") from exc
    except json.JSONDecodeError as exc:
        raise AIOSError(f"AI OS blueprint is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise AIOSError(f"AI OS blueprint could not be read: {exc}") from exc

    if not isinstance(payload, dict):
        raise AIOSError("AI OS blueprint root must be a JSON object.")
    return _parse_blueprint(payload)


def _resolve_blueprint_path(path: str | Path | None, config: Any | None) -> Path:
    if path is None:
        env_path = os.environ.get("JARVIS_AI_OS_BLUEPRINT", "").strip()
        if env_path:
            path = env_path
        elif config is not None:
            getter = getattr(config, "get_str", None)
            if callable(getter):
                path = getter("ai_os", "blueprint_file", fallback="config/ai_os.json")
            else:
                path = config.get("ai_os", "blueprint_file", fallback="config/ai_os.json")
        else:
            path = "config/ai_os.json"

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    from core.runtime.bootstrap import _resolve_path

    return _resolve_path(candidate)


def _parse_blueprint(payload: dict[str, Any]) -> SystemBlueprint:
    schema_version = _required_str(payload, "schema_version")
    product_name = _required_str(payload, "product_name")
    mission = _required_str(payload, "mission")

    philosophy = _required_mapping(payload, "philosophy")
    principles = [
        DesignPrinciple(
            id=_required_str(item, "id"),
            title=_required_str(item, "title"),
            summary=_required_str(item, "summary"),
            enforcement=_string_list(item.get("enforcement", []), "principle.enforcement"),
        )
        for item in _required_list(philosophy, "principles")
        if isinstance(item, dict)
    ]

    layers = [
        SystemLayer(
            id=_required_str(item, "id"),
            name=_required_str(item, "name"),
            responsibilities=_string_list(item.get("responsibilities", []), "layer.responsibilities"),
            interfaces=_string_list(item.get("interfaces", []), "layer.interfaces"),
            extensibility=_string_list(item.get("extensibility", []), "layer.extensibility"),
        )
        for item in _required_list(payload, "layers")
        if isinstance(item, dict)
    ]

    modes = [
        ModeDefinition(
            id=_required_str(item, "id"),
            name=_required_str(item, "name"),
            audience=_required_str(item, "audience"),
            features=_string_list(item.get("features", []), "mode.features"),
        )
        for item in _required_list(payload, "modes")
        if isinstance(item, dict)
    ]

    extensibility = _required_mapping(payload, "extensibility")
    ai_runtime = _required_mapping(payload, "ai_runtime")
    storage = _required_mapping(payload, "storage")
    security = _required_mapping(payload, "security")
    observability = _required_mapping(payload, "observability")
    future = _required_mapping(payload, "future")

    blueprint = SystemBlueprint(
        schema_version=schema_version,
        product_name=product_name,
        mission=mission,
        principles=principles,
        layers=layers,
        modes=modes,
        node_categories=_string_list(extensibility.get("workflow_node_categories", []), "workflow_node_categories"),
        provider_types=_string_list(ai_runtime.get("provider_types", []), "provider_types"),
        storage_backends=_string_list(storage.get("backends", []), "storage.backends"),
        security_controls=_string_list(security.get("controls", []), "security.controls"),
        observability=_string_list(observability.get("signals", []), "observability.signals"),
        future_extensions=_string_list(future.get("extensions", []), "future.extensions"),
        raw=payload,
    )
    _validate_blueprint(blueprint)
    return blueprint


def _validate_blueprint(blueprint: SystemBlueprint) -> None:
    if not blueprint.principles:
        raise AIOSError("AI OS blueprint must define at least one design principle.")
    if not blueprint.layers:
        raise AIOSError("AI OS blueprint must define at least one architecture layer.")
    if "local-first" not in blueprint.principle_ids:
        raise AIOSError("AI OS blueprint must include the local-first principle.")
    if "plugin-system" not in blueprint.layer_ids:
        raise AIOSError("AI OS blueprint must include a plugin-system layer.")
    if not blueprint.node_categories:
        raise AIOSError("AI OS blueprint must define workflow node categories.")


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise AIOSError(f"AI OS blueprint field '{key}' must be an object.")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise AIOSError(f"AI OS blueprint field '{key}' must be a list.")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise AIOSError(f"AI OS blueprint field '{key}' must be a non-empty string.")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AIOSError(f"AI OS blueprint field '{field_name}' must be a list of strings.")
    return [str(item).strip() for item in value if str(item).strip()]


__all__ = [
    "AIOSError",
    "DesignPrinciple",
    "ModeDefinition",
    "SystemBlueprint",
    "SystemLayer",
    "load_blueprint",
]
