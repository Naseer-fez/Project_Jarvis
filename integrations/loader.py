"""Auto-loader for integration clients under integrations/clients."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class IntegrationLoader:
    def load_all(self, config: Any, registry: Any) -> dict[str, list[str]]:
        del config  # Kept for callsite compatibility; integrations use env-only config.

        loaded: list[str] = []
        skipped: list[str] = []

        clients_dir = Path(__file__).parent / "clients"
        if not clients_dir.exists():
            return {"loaded": [], "skipped": ["clients/ dir not found"]}

        for py_file in sorted(clients_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = f"integrations.clients.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{py_file.stem} (import error: {exc})")
                logger.warning("Integration import failed %s: %s", py_file.stem, exc)
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls is BaseIntegration or not issubclass(cls, BaseIntegration):
                    continue
                if cls.__module__ != module_name:
                    continue

                try:
                    instance = cls()
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{cls.__name__} (init error: {exc})")
                    logger.warning("Integration init failed %s: %s", cls.__name__, exc)
                    continue

                try:
                    if bool(instance.is_available()):
                        registry.register(instance)
                        loaded.append(instance.name or cls.__name__)
                        logger.info("Integration loaded: %s", instance.name or cls.__name__)
                    else:
                        skipped.append(f"{cls.__name__} (not available)")
                        logger.debug("Integration skipped: %s", cls.__name__)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{cls.__name__} (availability/register error: {exc})")
                    logger.warning("Integration registration failed %s: %s", cls.__name__, exc)

        return {"loaded": loaded, "skipped": skipped}


def load_all(config: Any, registry: Any) -> dict[str, list[str]]:
    """Backward-compatible function wrapper for older callsites."""
    return IntegrationLoader().load_all(config=config, registry=registry)


__all__ = ["IntegrationLoader", "load_all"]
