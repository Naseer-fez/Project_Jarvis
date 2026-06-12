# Analysis Report for vision.py

## Dependencies
- __future__.annotations
- logging
- pathlib.Path
- typing.Any

## Schemas
- VisionTool

## API Contracts
- VisionTool.__init__(self, config)
- VisionTool._get(self, key, default)
- VisionTool.analyze(self, image_path, prompt)
- VisionTool._call_llava(self, image_path, prompt)

## Configuration Variables
- _SUPPORTED_EXTENSIONS

## Assumptions & Notes
- Module Docstring: core/tools/vision.py — Vision (image analysis) tool.

Wraps Ollama's LLaVA model for image understanding.
Tests can patch _call_llava() to inject fake responses.

