"""Auto-discovery loader for integrations/clients plugins."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration
from integrations.registry import IntegrationRegistry

logger = logging.getLogger(__name__)


def _iter_client_modules() -> list[str]:
    clients_dir = Path(__file__).resolve().parent / "clients"
    if not clients_dir.exists():
        return []

    modules: list[str] = []
    for file in sorted(clients_dir.glob("*.py")):
        if file.name == "__init__.py":
            continue
        modules.append(f"integrations.clients.{file.stem}")
    return modules


def _instantiate(cls: type[BaseIntegration], config: Any) -> BaseIntegration:
    """Instantiate integration with best-effort constructor compatibility."""
    for call in (
        lambda: cls(config=config),
        lambda: cls(config),
        lambda: cls(),
    ):
        try:
            return call()
        except TypeError:
            continue
    return cls()


def load_all(config: Any, registry: IntegrationRegistry) -> dict[str, list[str]]:
    """Discover and load all integration plugins from integrations/clients."""
    result = {"loaded": [], "skipped": [], "errors": []}

    for module_path in _iter_client_modules():
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning("Integration module skipped (import error): %s (%s)", module_path, exc)
            result["skipped"].append(module_path)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("Integration module failed to load: %s", module_path)
            result["errors"].append(f"{module_path}: {exc}")
            continue

        classes = [
            member
            for _, member in inspect.getmembers(module, inspect.isclass)
            if issubclass(member, BaseIntegration)
            and member is not BaseIntegration
            and member.__module__ == module.__name__
        ]

        if not classes:
            logger.info("Integration module loaded with no BaseIntegration subclasses: %s", module_path)
            result["skipped"].append(module_path)
            continue

        for cls in classes:
            integration_name = f"{module_path}.{cls.__name__}"
            try:
                instance = _instantiate(cls, config)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Integration instantiation failed: %s", integration_name)
                result["errors"].append(f"{integration_name}: {exc}")
                continue

            try:
                available = bool(instance.is_available())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Integration availability check failed: %s (%s)", integration_name, exc)
                result["skipped"].append(integration_name)
                continue

            if not available:
                logger.warning("Integration skipped (unavailable): %s", integration_name)
                result["skipped"].append(integration_name)
                continue

            try:
                registry.register(instance)
                logger.info("Integration loaded: %s", integration_name)
                result["loaded"].append(instance.name or cls.__name__)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Integration registration failed: %s", integration_name)
                result["errors"].append(f"{integration_name}: {exc}")

    logger.info(
        "Integration load summary: loaded=%d skipped=%d errors=%d",
        len(result["loaded"]),
        len(result["skipped"]),
        len(result["errors"]),
    )
    return result


__all__ = ["load_all"]
