# API Analyst Report: runtime\container.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `from typing import Any`
- `from typing import Callable`
- `from typing import Dict`
- `from typing import Tuple`
- `from typing import Type`
- `from typing import Union`

## Schemas & API Contracts (Classes)

### Class `ServiceContainer`
> Lightweight, thread-safe Dependency Injection (DI) Container.
Enables components to be decoupled and custom subclasses/mock implementations
to be dynamically registered and resolved.

**Methods:**
- `def __init__(self) -> None`
- `def register(self, name: str, factory_or_class: Union[Type, Callable], is_singleton: bool=True) -> None`
  - *Register a class or a factory function for a service.*
- `def register_instance(self, name: str, instance: Any) -> None`
  - *Register a pre-constructed instance of a service.*
- `def has(self, name: str) -> bool`
  - *Check if a service is registered in the container.*
- `def resolve(self, name: str, **kwargs) -> Any`
  - *Resolve a service instance. If registered as a singleton, the cached instance*
- `def reset(self) -> None`
  - *Clears all registered providers and cached instances.*

