# API Analyst Report: tools\vision.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `from pathlib import Path`
- `from typing import Any`

## Configuration Variables
- `_SUPPORTED_EXTENSIONS` = `{'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}`

## Schemas & API Contracts (Classes)

### Class `VisionTool`
> Analyze images using LLaVA or a test stub.

**Methods:**
- `def __init__(self, config: Any) -> None`
- `def _get(self, key: str, default: str) -> str`
- `def analyze(self, image_path: str, prompt: str='Describe this image.') -> str`
  - *Analyze *image_path* using LLaVA and return a text description.*
- `def _call_llava(self, image_path: str, prompt: str) -> str`
  - *Call the LLaVA model. Override in tests.*

