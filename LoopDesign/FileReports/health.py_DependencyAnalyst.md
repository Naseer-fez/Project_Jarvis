# Dependency Analysis Report for introspection\health.py

## Library Requirements
- from __future__ import annotations
- from core.ops.production import is_production
- from core.ops.production import validate_production_config
- from core.runtime.import_validator import StartupValidator
- from core.runtime.paths import _resolve_path
- from dataclasses import dataclass
- from dataclasses import field
- from enum import Enum
- from pathlib import Path
- from urllib.request import urlopen
- import importlib.util
- import os
- import subprocess
- import sys

## Service Dependencies
- URL: http://localhost:11434

## Hidden Execution Links
- subprocess.run

## Configurations / Assumptions
- Analyzed via AST and regex.
