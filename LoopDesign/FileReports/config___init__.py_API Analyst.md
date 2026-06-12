# API Analyst Report: config\__init__.py

## Dependencies
- `from __future__ import annotations`
- `import configparser`
- `import os`

## Schemas & API Contracts (Classes)

### Class `JarvisConfig(configparser.ConfigParser)`
> Typed config manager that inherits from configparser.ConfigParser
to maintain full backward compatibility while exposing typed accessors,
standardizing env-var overrides, and providing unified fallback lookups.

**Methods:**
- `def get_str(self, section: str, key: str, fallback: str='') -> str`
  - *Get config string, checking env-var overrides first.*
- `def get_bool(self, section: str, key: str, fallback: bool=False) -> bool`
  - *Get config boolean, checking env-var overrides first.*
- `def get_int(self, section: str, key: str, fallback: int=0) -> int`
  - *Get config integer, checking env-var overrides first.*


## Functions & Endpoints

### `load_config`
`def load_config(config_path: str) -> JarvisConfig`
> Load INI config from an absolute path or relative to PROJECT_ROOT
into JarvisConfig, with env-var resolution.
