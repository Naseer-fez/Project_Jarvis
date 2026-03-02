"""Auto-loader for integration clients under integrations/clients."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration
from integrations.registry import IntegrationRegistry

logger = logging.getLogger(__name__)


def _discover_modules() -> list[str]:
    clients_dir = Path(__file__).resolve().parent / "clients"
    if not clients_dir.exists():
        return []

    modules: list[str] = []
    for file_path in sorted(clients_dir.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        modules.append(f"integrations.clients.{file_path.stem}")
    return modules


def _integration_classes(module: Any) -> list[type[BaseIntegration]]:
    classes: list[type[BaseIntegration]] = []
    for _, member in inspect.getmembers(module, inspect.isclass):
        if member is BaseIntegration:
            continue
        if not issubclass(member, BaseIntegration):
            continue
        if member.__module__ != module.__name__:
            continue
        classes.append(member)
    return classes


def _build_instance(cls: type[BaseIntegration], config: Any) -> BaseIntegration:
    for ctor in (
        lambda: cls(config=config),
        lambda: cls(config),
        lambda: cls(),
    ):
        try:
            return ctor()
        except TypeError:
            continue
    return cls()


def load_all(config: Any, registry: IntegrationRegistry) -> dict[str, list[str]]:
    """Discover all clients and register available integrations."""
    summary: dict[str, list[str]] = {
        "loaded": [],
        "skipped": [],
        "errors": [],
    }

    for module_name in _discover_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Integration module import failed: %s (%s)", module_name, exc)
            summary["errors"].append(f"{module_name}: {exc}")
            continue

        classes = _integration_classes(module)
        if not classes:
            logger.info("No BaseIntegration subclass found in %s", module_name)
            summary["skipped"].append(f"{module_name}: no integration class")
            continue

        for cls in classes:
            fqcn = f"{module_name}.{cls.__name__}"
            try:
                instance = _build_instance(cls, config)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Integration instantiation failed: %s (%s)", fqcn, exc)
                summary["errors"].append(f"{fqcn}: {exc}")
                continue

            try:
                available = bool(instance.is_available())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Availability check failed: %s (%s)", fqcn, exc)
                summary["errors"].append(f"{fqcn}: {exc}")
                continue

            if not available:
                reason = (getattr(instance, "unavailable_reason", "") or "unavailable").strip()
                logger.info("Integration skipped: %s (%s)", fqcn, reason)
                summary["skipped"].append(f"{fqcn}: {reason}")
                continue

            try:
                registry.register(instance)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Integration register failed: %s (%s)", fqcn, exc)
                summary["errors"].append(f"{fqcn}: {exc}")
                continue

            loaded_name = instance.name or cls.__name__
            summary["loaded"].append(loaded_name)
            logger.info("Integration loaded: %s", loaded_name)

    logger.info(
        "Integration load complete | loaded=%d skipped=%d errors=%d",
        len(summary["loaded"]),
        len(summary["skipped"]),
        len(summary["errors"]),
    )
    return summary


__all__ = ["load_all"]
