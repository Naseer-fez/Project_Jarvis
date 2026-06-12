# API Analyst Report: llm\model_spec.py

## Dependencies
- `from __future__ import annotations`
- `import configparser`
- `import logging`
- `from dataclasses import dataclass`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `ModelSpec`
> Immutable descriptor for a single LLM model.

**Fields/Schema:**
  - `name: str`
  - `provider: str`
  - `tier: int`
  - `weight: float`
  - `max_context_tokens: int`
  - `supports_tools: bool`
  - `supports_vision: bool`
  - `latency_class: str`
  - `reasoning_capability: float`



### Class `RoutingDecision`
> Result of an adaptive routing decision — carries the chosen model + rationale.

**Fields/Schema:**
  - `model: str`
  - `provider: str`
  - `tier: int`
  - `reason: str`
  - `weight: float`
  - `fallback_model: str | None`
  - `fallback_provider: str | None`



### Class `ModelRegistry`
> Queryable catalog of all known model specs.

Loads built-in defaults, then overlays any ``[models.registry]`` section
from the Jarvis config file.

**Methods:**
- `def __init__(self, config: configparser.ConfigParser | None=None) -> None`
- `def get(self, model_name: str) -> ModelSpec | None`
  - *Look up a spec by exact name, or by family prefix.*
- `def get_tier(self, model_name: str) -> int`
  - *Return the tier for a model, defaulting to 2.*
- `def get_weight(self, model_name: str) -> float`
  - *Return the cost weight for a model, defaulting to 0.5.*
- `def all_specs(self) -> list[ModelSpec]`
- `def by_provider(self, provider: str) -> list[ModelSpec]`
- `def by_tier(self, tier: int) -> list[ModelSpec]`
- `def get_cheapest_capable(self, *, min_reasoning: float=0.0, min_context: int=0, needs_tools: bool=False, needs_vision: bool=False, available_models: set[str] | None=None, providers: set[str] | None=None) -> list[ModelSpec]`
  - *Return models meeting the requirements, sorted cheapest-first.*
- `def register(self, spec: ModelSpec) -> None`
  - *Add or replace a model spec.*
- `def _load_config_overrides(self, config: configparser.ConfigParser) -> None`
  - *Parse ``[models.registry]`` entries like ``mistral:7b = tier=2,weight=0.10``.*
- @staticmethod
- `def _parse_override(raw: str) -> dict[str, Any]`
  - *Parse ``tier=2,weight=0.10,supports_tools=true`` into a dict.*

