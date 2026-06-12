# Documentation Report: loader.py

## Assumptions
- Integrations reside in `integrations.clients` package
- Any `.py` file not prefixed with `_` is a potential integration module.
- It iterates through all modules inside `clients/`, uses `importlib` to import them, finds subclasses of `BaseIntegration`, and instantiates them.
- Availability is checked before registering in `registry`.
- Exceptions during import, instantiation, or registration are caught, logged, and skipped to allow gracefully failing.
- Callers may pass a config, but integrations currently use environment variables.

## Schema / API Contract
- `IntegrationLoader.load_all(config: Any, registry: Any) -> dict[str, list[str]]` where returned dictionary has `loaded` and `skipped` lists of integration names.
- `load_all` top-level function wraps `IntegrationLoader().load_all`.

## Dependencies
- `importlib`, `inspect`, `logging`, `pathlib`, `typing`
- `integrations.base.BaseIntegration`

## Configuration Variables
None explicitly referenced, assumes integrations load their own.

## Prompts
None.
