# loader.py API Analyst Report

## Overview
Auto-loader for dynamically discovering and loading integration clients from the `clients/` directory.

## API Contracts & Methods
- `IntegrationLoader.load_all(config: Any, registry: Any) -> dict[str, list[str]]`
  - Returns a dictionary with `loaded` and `skipped` lists.
  - Scans `clients/*.py`.
  - Instantiates subclasses of `BaseIntegration`.
  - Calls `instance.is_available()`. If true, registers the instance in `registry`.
- `load_all(config: Any, registry: Any) -> dict[str, list[str]]`: Backwards-compatible wrapper.

## Assumptions
- `clients` directory is adjacent to `loader.py`.
- Integrations must use env-only config (`config` argument is ignored/deleted).
- Skips files starting with `_`.

## Dependencies
- `integrations.base.BaseIntegration`

## Prompts
- None.
