# `script.py` and `script_ast.py` - API Analyst Report

## Overview
These scripts appear to be automated analysis tools (written by a Configuration Analyst or similar agent) to parse the `integrations` directory, extract AST data, and generate markdown reports / extract prompts.

## Endpoints / Tools
- They do not expose Jarvis tools/endpoints.
- They execute locally to generate `_Configuration Analyst.md` reports.

## External Contracts / Dependencies
- Relies on standard library `ast`, `os`, `json`, `re`.

## Assumptions
- Uses heuristics like searching for "you are", "prompt", "instruction" in strings or docstrings to extract prompts.
- Hardcodes the output paths to `d:\AI\Jarvis\LoopDesign\FileReports` and `d:\AI\Jarvis\LoopDesign\Prompts`.
