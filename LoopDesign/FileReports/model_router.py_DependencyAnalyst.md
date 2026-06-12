# Dependency Analysis Report for llm\model_router.py

## Library Requirements
- from __future__ import annotations
- from core.config.defaults import OLLAMA_BASE_URL
- from core.llm.model_spec import ModelRegistry
- from core.llm.model_spec import RoutingDecision
- from core.llm.ollama_client import list_models_sync
- from typing import Any
- from typing import Iterable
- import asyncio
- import logging
- import os
- import threading
- import time

## Service Dependencies
- asyncio.get_running_loop

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
