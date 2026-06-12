# File Report: vision.py
**Role**: Prompt Recovery Specialist

## Dependencies
- requests
- typing
- logging
- base64
- __future__
- pathlib

## Configuration Variables & Constants

## Schemas & API Contracts
### Class `VisionTool`
**Assumptions/Doc**: Analyze images using LLaVA or a test stub.
**Methods**: __init__, _get, analyze, _call_llava

### Function `__init__`
**Args**: self, config

### Function `_get`
**Args**: self, key, default

### Function `analyze`
**Args**: self, image_path, prompt
**Assumptions/Doc**: Analyze *image_path* using LLaVA and return a text description.

Raises:
    FileNotFoundError: if the image file does not exist.
    ValueError: if the file extension is not a supported image format.

### Function `_call_llava`
**Args**: self, image_path, prompt
**Assumptions/Doc**: Call the LLaVA model. Override in tests.

## Prompts and LLM Directives
No explicit prompts found in module scope.
