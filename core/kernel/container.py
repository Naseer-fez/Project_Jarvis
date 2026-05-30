"""
core/kernel/container.py
────────────────────────
Decoupled Dependency Injection container supporting global singletons and
task-scoped resolutions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple, Type, Union

logger = logging.getLogger("Jarvis.Kernel.Container")


class ServiceContainer:
    """
    Lightweight, thread-safe Dependency Injection (DI) Container.
    Enables components to be decoupled and custom subclasses/mock implementations
    to be dynamically registered and resolved.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, Tuple[Union[Type, Callable], bool]] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, name: str, factory_or_class: Union[Type, Callable], is_singleton: bool = True) -> None:
        """Register a class or a factory function for a service."""
        key = name.strip().lower()
        self._providers[key] = (factory_or_class, is_singleton)
        logger.debug("Registered service '%s' (singleton=%s)", key, is_singleton)
        # Invalidate any cached instance if re-registered
        if key in self._instances:
            del self._instances[key]

    def register_instance(self, name: str, instance: Any) -> None:
        """Register a pre-constructed instance of a service."""
        key = name.strip().lower()
        self._instances[key] = instance
        logger.debug("Registered instance for service '%s'", key)

    def has(self, name: str) -> bool:
        """Check if a service is registered in the container."""
        key = name.strip().lower()
        return key in self._instances or key in self._providers

    def resolve(self, name: str, **kwargs) -> Any:
        """
        Resolve a service instance. If registered as a singleton, the cached instance
        is returned; otherwise, a fresh instance is constructed.
        """
        key = name.strip().lower()
        if key in self._instances:
            return self._instances[key]

        if key not in self._providers:
            raise ValueError(f"Service '{name}' is not registered in the container.")

        factory_or_class, is_singleton = self._providers[key]

        try:
            import inspect
            if isinstance(factory_or_class, type):
                try:
                    sig = inspect.signature(factory_or_class.__init__)
                    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if has_var_keyword:
                        valid_kwargs = kwargs
                    else:
                        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                except Exception:
                    valid_kwargs = kwargs
                instance = factory_or_class(**valid_kwargs)
            elif callable(factory_or_class):
                try:
                    sig = inspect.signature(factory_or_class)
                    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if has_var_keyword:
                        valid_kwargs = kwargs
                    else:
                        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                except Exception:
                    valid_kwargs = {}
                instance = factory_or_class(**valid_kwargs)
            else:
                instance = factory_or_class
        except Exception as e:
            logger.exception("Failed to resolve service '%s'", name)
            raise RuntimeError(f"Failed to instantiate service '{name}': {e}") from e

        if is_singleton:
            self._instances[key] = instance

        return instance

    def reset(self) -> None:
        """Clears all registered providers and cached instances."""
        self._providers.clear()
        self._instances.clear()
        logger.debug("Container reset.")
