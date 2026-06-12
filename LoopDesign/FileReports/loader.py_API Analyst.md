# `loader.py` - API Analyst Report

## Overview
Auto-loader script for all integration clients placed under the `integrations/clients/` folder. It discovers and initializes them dynamically.

## Endpoints / Tools
- `load_all(config: Any, registry: Any) -> dict[str, list[str]]`: Discovers and loads client integrations. Calls `registry.register` on available integrations.

## External Contracts / Dependencies
- Loads from `integrations.clients.*`.
- Uses `integrations.base.BaseIntegration` to verify if classes are valid integrations.
- Backward compatibility wrapper `load_all` is preserved for older callsites.

## Assumptions
- Ignores python files starting with `_` (e.g. `__init__.py`).
- Iterates over all module classes, only checks for `issubclass(cls, BaseIntegration)`.
- Fails safely on import error, initialization error, or availability evaluation errors, registering them to a `skipped` list.
- Assumes `config` parameter is no longer explicitly needed but kept for signature compatibility. Integrations use env-only configuration.
