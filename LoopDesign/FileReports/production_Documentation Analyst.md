# Analysis Report for production.py

## Dependencies
- __future__.annotations
- os
- dataclasses.dataclass
- dataclasses.field
- typing.Any

## Schemas
- ProductionCheck
- ProductionCheck attribute: errors
- ProductionCheck attribute: warnings

## API Contracts
- ProductionCheck.ok(self)
- _get(config, section, key, fallback)
- _get_bool(config, section, key, fallback)
- is_production(config)
- validate_production_config(config)

## Configuration Variables
- PUBLIC_HOSTS
- DANGEROUS_ENV_FLAGS

## Assumptions & Notes
None

