# Analysis Report for model_spec.py

## Dependencies
- __future__.annotations
- configparser
- logging
- dataclasses.dataclass
- typing.Any

## Schemas
- ModelSpec
- ModelSpec attribute: name
- ModelSpec attribute: provider
- ModelSpec attribute: tier
- ModelSpec attribute: weight
- ModelSpec attribute: max_context_tokens
- ModelSpec attribute: supports_tools
- ModelSpec attribute: supports_vision
- ModelSpec attribute: latency_class
- ModelSpec attribute: reasoning_capability
- RoutingDecision
- RoutingDecision attribute: model
- RoutingDecision attribute: provider
- RoutingDecision attribute: tier
- RoutingDecision attribute: reason
- RoutingDecision attribute: weight
- RoutingDecision attribute: fallback_model
- RoutingDecision attribute: fallback_provider
- ModelRegistry

## API Contracts
- ModelRegistry.__init__(self, config)
- ModelRegistry.get(self, model_name)
- ModelRegistry.get_tier(self, model_name)
- ModelRegistry.get_weight(self, model_name)
- ModelRegistry.all_specs(self)
- ModelRegistry.by_provider(self, provider)
- ModelRegistry.by_tier(self, tier)
- ModelRegistry.get_cheapest_capable(self)
- ModelRegistry.register(self, spec)
- ModelRegistry._load_config_overrides(self, config)
- ModelRegistry._parse_override(raw)

## Configuration Variables
- _BUILTIN_SPECS (typed)

## Assumptions & Notes
- Module Docstring: Model specification registry for adaptive routing.

Each model known to Jarvis is described by a ``ModelSpec`` — a frozen record
of its cost, capabilities and constraints.  ``ModelRegistry`` aggregates
specs and exposes query helpers used by the adaptive ``ModelRouter``.

