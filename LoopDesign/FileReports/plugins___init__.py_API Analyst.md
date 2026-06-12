# API Analyst Report: plugins\__init__.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `from pathlib import Path`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `PluginCatalog`
> Mock/Stub of PluginCatalog to support the Dashboard summary API after manifest removal.

**Methods:**
- `def __init__(self, root: str | Path | None=None, config: Any | None=None, enabled_scopes: set[str] | None=None) -> None`
- `def summary(self) -> dict[str, Any]`


### Class `PluginManifest`


### Class `PluginManifestError(ValueError)`


## Functions & Endpoints

### `load_plugin_manifest`
`def load_plugin_manifest(plugin_dir: Any) -> Any`