"""JSON workflow template catalog and validator."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.workflow.engine import WorkflowStep


class WorkflowCatalogError(ValueError):
    """Raised when a workflow template is malformed."""


_WORKFLOW_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_PLACEHOLDER_PREFIXES = ("${", "{{")
_URL_MARKERS = ("://", "localhost:", "127.0.0.1:")


@dataclass(frozen=True)
class WorkflowTemplate:
    workflow_id: str
    name: str
    version: str
    description: str
    mode: str
    tags: list[str]
    trigger: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
    permissions: list[dict[str, str]]
    marketplace: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    path: Path | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def to_steps(self) -> list[WorkflowStep]:
        steps: list[WorkflowStep] = []
        for node in self.nodes:
            tool_name = str(node.get("tool") or node.get("type") or "").strip()
            if not tool_name:
                continue
            depends_on = [
                edge["from"]
                for edge in self.edges
                if edge.get("to") == node.get("id") and edge.get("from")
            ]
            steps.append(
                WorkflowStep(
                    tool_name=tool_name,
                    args=dict(node.get("config", {}) or {}),
                    step_id=str(node.get("id") or tool_name),
                    retry_count=int(node.get("retry_count", 1)),
                    timeout=float(node.get("timeout", 30.0)),
                    depends_on=depends_on,
                )
            )
        return steps


class WorkflowCatalog:
    def __init__(self, root: str | Path | None = None, *, config: Any | None = None) -> None:
        self.root = _resolve_workflow_root(root=root, config=config)
        self._templates: dict[str, WorkflowTemplate] = {}
        self.errors: dict[str, str] = {}

    def refresh(self) -> list[WorkflowTemplate]:
        self._templates.clear()
        self.errors.clear()
        if not self.root.exists() or not self.root.is_dir():
            return []

        for path in sorted(self.root.glob("*.workflow.json")):
            try:
                template = load_workflow_template(path)
            except WorkflowCatalogError as exc:
                self.errors[str(path)] = str(exc)
                continue
            self._templates[template.workflow_id] = template
        return self.list_templates()

    def list_templates(self) -> list[WorkflowTemplate]:
        if not self._templates and not self.errors:
            self.refresh()
        return [self._templates[key] for key in sorted(self._templates)]

    def get(self, workflow_id: str) -> WorkflowTemplate | None:
        if not self._templates and not self.errors:
            self.refresh()
        return self._templates.get(workflow_id)

    def summary(self) -> dict[str, Any]:
        templates = self.list_templates()
        return {
            "root": str(self.root),
            "count": len(templates),
            "templates": [
                {
                    "id": template.workflow_id,
                    "name": template.name,
                    "version": template.version,
                    "mode": template.mode,
                    "tags": template.tags,
                    "node_count": len(template.nodes),
                    "permission_scopes": [permission.get("scope", "") for permission in template.permissions],
                    "one_click_install": bool(template.marketplace.get("one_click_install", True)),
                }
                for template in templates
            ],
            "errors": dict(self.errors),
        }


def load_workflow_template(path: str | Path) -> WorkflowTemplate:
    workflow_path = Path(path)
    try:
        with workflow_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise WorkflowCatalogError(f"Workflow template not found: {workflow_path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowCatalogError(f"Workflow template is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise WorkflowCatalogError(f"Workflow template could not be read: {exc}") from exc

    if not isinstance(payload, dict):
        raise WorkflowCatalogError("Workflow template root must be a JSON object.")
    return _parse_template(payload, path=workflow_path)


def _parse_template(payload: dict[str, Any], *, path: Path | None = None) -> WorkflowTemplate:
    workflow_id = _required_str(payload, "id")
    if not _WORKFLOW_ID_RE.match(workflow_id):
        raise WorkflowCatalogError("Workflow id must be lowercase and URL-safe.")

    nodes = _required_list(payload, "nodes")
    edges = payload.get("edges", [])
    permissions = payload.get("permissions", [])
    if not isinstance(edges, list):
        raise WorkflowCatalogError("Workflow edges must be a list.")
    if not isinstance(permissions, list):
        raise WorkflowCatalogError("Workflow permissions must be a list.")

    _validate_nodes(nodes)
    _validate_edges(nodes, edges)
    _validate_no_environment_specific_values(payload)

    return WorkflowTemplate(
        workflow_id=workflow_id,
        name=_required_str(payload, "name"),
        version=_required_str(payload, "version"),
        description=str(payload.get("description", "")).strip(),
        mode=str(payload.get("mode", "beginner")).strip() or "beginner",
        tags=_string_list(payload.get("tags", []), "tags"),
        trigger=dict(payload.get("trigger", {}) or {}),
        nodes=[dict(node) for node in nodes if isinstance(node, dict)],
        edges=[_edge_to_dict(edge) for edge in edges],
        permissions=[dict(permission) for permission in permissions if isinstance(permission, dict)],
        marketplace=dict(payload.get("marketplace", {}) or {}),
        raw=payload,
        path=path,
    )


def _validate_nodes(nodes: list[Any]) -> None:
    if not nodes:
        raise WorkflowCatalogError("Workflow template must contain at least one node.")

    seen: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise WorkflowCatalogError("Workflow nodes must be objects.")
        node_id = _required_str(node, "id")
        if node_id in seen:
            raise WorkflowCatalogError(f"Duplicate workflow node id: {node_id}")
        seen.add(node_id)
        _required_str(node, "type")
        _required_str(node, "name")
        config = node.get("config", {})
        if config is not None and not isinstance(config, dict):
            raise WorkflowCatalogError(f"Workflow node '{node_id}' config must be an object.")


def _validate_edges(nodes: list[Any], edges: list[Any]) -> None:
    node_ids = {str(node.get("id", "")).strip() for node in nodes if isinstance(node, dict)}
    for edge in edges:
        normalized = _edge_to_dict(edge)
        source = normalized.get("from", "")
        target = normalized.get("to", "")
        if source not in node_ids or target not in node_ids:
            raise WorkflowCatalogError(f"Workflow edge references missing node: {source} -> {target}")


def _edge_to_dict(edge: Any) -> dict[str, str]:
    if not isinstance(edge, dict):
        raise WorkflowCatalogError("Workflow edges must be objects.")
    source = str(edge.get("from", "")).strip()
    target = str(edge.get("to", "")).strip()
    if not source or not target:
        raise WorkflowCatalogError("Workflow edges require non-empty 'from' and 'to'.")
    return {"from": source, "to": target}


def _validate_no_environment_specific_values(value: Any, path: str = "workflow") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_no_environment_specific_values(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_environment_specific_values(child, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return

    text = value.strip()
    if not text or text.startswith(_PLACEHOLDER_PREFIXES):
        return
    if Path(text).is_absolute() or any(marker in text.lower() for marker in _URL_MARKERS):
        raise WorkflowCatalogError(
            f"Workflow template contains environment-specific value at {path}; use a placeholder or config key."
        )


def _resolve_workflow_root(root: str | Path | None, config: Any | None) -> Path:
    if root is None:
        env_root = os.environ.get("JARVIS_WORKFLOW_CATALOG_DIR", "").strip()
        if env_root:
            root = env_root
        elif config is not None:
            getter = getattr(config, "get_str", None)
            if callable(getter):
                root = getter("ai_os", "workflow_catalog_dir", fallback="workflows/templates")
            else:
                root = config.get("ai_os", "workflow_catalog_dir", fallback="workflows/templates")
        else:
            root = "workflows/templates"

    candidate = Path(root)
    if candidate.is_absolute():
        return candidate

    from core.runtime.bootstrap import _resolve_path

    return _resolve_path(candidate)


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise WorkflowCatalogError(f"Workflow template field '{key}' must be a list.")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise WorkflowCatalogError(f"Workflow template field '{key}' must be a non-empty string.")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise WorkflowCatalogError(f"Workflow template field '{field_name}' must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


__all__ = [
    "WorkflowCatalog",
    "WorkflowCatalogError",
    "WorkflowTemplate",
    "load_workflow_template",
]
