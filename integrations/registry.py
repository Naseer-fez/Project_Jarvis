"""
integrations/api_registry.py

Central registry for all Jarvis external-API integrations.

How it works
------------
1.  Each integration subclasses BaseIntegration and is imported here.
2.  ``register()`` validates the class and adds it to the in-memory registry.
3.  ``tool_router.py`` calls ``get_tool(name)`` or ``list_schemas()`` — zero
    changes needed to core/ to add a new API.

Adding a new integration
------------------------
    # In your new module:
    class MyApiIntegration(BaseIntegration):
        tool_name  = "my_api_action"
        risk_level = RiskLevel.READ_ONLY
        ...

    # Then here:
    from integrations.my_api.client import MyApiIntegration
    register(MyApiIntegration)
"""

from __future__ import annotations

import logging
from typing import Type

from integrations.base_integration import BaseIntegration, RiskLevel

# ---------------------------------------------------------------------------
# Try to import core logger; fall back gracefully if running standalone
# ---------------------------------------------------------------------------
try:
    from core.logger import get_logger          # type: ignore
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry store
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BaseIntegration] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(cls: Type[BaseIntegration]) -> None:
    """
    Instantiate and register an integration class.

    Raises
    ------
    TypeError  – if cls doesn't subclass BaseIntegration.
    ValueError – if tool_name or risk_level are not set.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseIntegration)):
        raise TypeError(f"{cls} must subclass BaseIntegration")

    instance = cls()

    if instance.tool_name is NotImplemented or not instance.tool_name:
        raise ValueError(f"{cls.__name__} must define a non-empty tool_name")

    if instance.risk_level is NotImplemented:
        raise ValueError(f"{cls.__name__} must define a risk_level")

    if instance.tool_name in _REGISTRY:
        logger.warning(
            "api_registry: overwriting existing tool '%s'", instance.tool_name
        )

    _REGISTRY[instance.tool_name] = instance
    logger.info(
        "api_registry: registered tool '%s' [%s]",
        instance.tool_name,
        instance.risk_level.value,
    )


def get_tool(name: str) -> BaseIntegration | None:
    """Return the integration instance for *name*, or None if not found."""
    return _REGISTRY.get(name)


def list_schemas() -> list[dict]:
    """
    Return all tool schemas — used by task_planner to populate the LLM's
    available-tools context.
    """
    return [tool.tool_schema for tool in _REGISTRY.values()]


def list_tools() -> dict[str, str]:
    """Return {tool_name: risk_level} for every registered integration."""
    return {name: tool.risk_level.value for name, tool in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# Auto-registration — import integrations here so they self-register on import
# ---------------------------------------------------------------------------

def _load_integrations() -> None:
    """
    Import all integration modules.  Failures are logged but do NOT crash
    Jarvis — an unavailable plugin should degrade gracefully.
    """
    _plugins = [
        ("integrations.weather_api.client", "WeatherIntegration"),
        # ("integrations.custom_api_2.client", "CustomApi2Integration"),  # add next plugin here
    ]

    for module_path, class_name in _plugins:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            register(cls)
        except ImportError as exc:
            logger.warning(
                "api_registry: could not import '%s.%s' — %s",
                module_path,
                class_name,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "api_registry: failed to register '%s.%s' — %s",
                module_path,
                class_name,
                exc,
            )


_load_integrations()
