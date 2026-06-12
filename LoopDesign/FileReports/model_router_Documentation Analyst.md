# Analysis Report for model_router.py

## Dependencies
- __future__.annotations
- asyncio
- logging
- os
- threading
- time
- typing.Iterable
- typing.Any
- core.config.defaults.OLLAMA_BASE_URL
- core.llm.ollama_client.list_models_sync
- core.llm.model_spec.ModelRegistry
- core.llm.model_spec.RoutingDecision

## Schemas
- ModelRouter

## API Contracts
- ModelRouter.__init__(self, config)
- ModelRouter.set_telemetry(self, telemetry)
- ModelRouter.registry(self)
- ModelRouter.strategy(self)
- ModelRouter.route(self, task_type)
- ModelRouter.escalate(self, current_model)
- ModelRouter.pick_model(self, task_type)
- ModelRouter.get_best_available(self, task_type)
- ModelRouter.route_adaptive(self, classification)
- ModelRouter.should_escalate(self, model, task_type, response)
- ModelRouter._pick_model_from_tier(self, tier)
- ModelRouter.is_available(self, model_name)
- ModelRouter.list_available(self)
- ModelRouter.refresh_available_models(self, base_url)
- ModelRouter.list_available_without_refresh(self)
- ModelRouter.set_available_models(self, models)
- ModelRouter._cfg(self, key, default)
- ModelRouter._ollama_base_url(self)
- ModelRouter._resolve_available_variant(self, model_name)

## Configuration Variables
- _MODEL_DISCOVERY_TTL_S
- _REASONING_BY_COMPLEXITY

## Assumptions & Notes
- Module Docstring: Intelligent multi-stage model router for Jarvis.

Supports two strategies:
  - ``adaptive`` (default): cost-minimising selection based on task
    classification signals, model capabilities, availability, and
    historical reliability from telemetry.
  - ``static``: legacy tier-lookup behaviour (backward compatible).

Public surface (all preserved from the original):
  - ``route(task_type)``
  - ``pick_model(task_type)``
  - ``get_best_available(task_type)``
  - ``escalate(current_model)``

New:
  - ``route_adaptive(classification)`` → ``RoutingDecision``
  - ``should_escalate(model, task_type, response)`` → bool

