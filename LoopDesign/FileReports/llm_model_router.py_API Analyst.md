# API Analyst Report: llm\model_router.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import os`
- `import threading`
- `import time`
- `from typing import Iterable`
- `from typing import Any`
- `from core.config.defaults import OLLAMA_BASE_URL`
- `from core.llm.ollama_client import list_models_sync`
- `from core.llm.model_spec import ModelRegistry`
- `from core.llm.model_spec import RoutingDecision`

## Configuration Variables
- `_MODEL_DISCOVERY_TTL_S` = `30.0`
- `_REASONING_BY_COMPLEXITY` = `[(0.15, 0.1), (0.4, 0.25), (0.6, 0.4), (0.8, 0.6), (1.01, 0.75)]`
- `TIERS` = `{1: ['llama3.2:1b', 'qwen2.5:0.5b', 'qwen2.5:1.5b', 'gemma2:2b'], 2: ['mistral:7b', 'llama3:8b', 'qwen2.5:7b'], 3: ['deepseek-r1:8b', 'deepseek-r1:14b', 'llama3.3:70b']}`
- `TASK_TIERS` = `{'intent': 1, 'intent_classification': 1, 'reflex': 1, 'web_search_summary': 1, 'tool_parameter_extraction': 2, 'context_title_generation': 1, 'synthesis': 1, 'summarize': 1, 'memory_summarization': 1, 'context_summarization': 1, 'chat': 2, 'general': 2, 'planning': 3, 'plan': 3, 'tool_selection': 2, 'tool_picker': 2, 'reflection': 2, 'deep_reasoning': 3, 'complex_debugging': 3, 'architecture_generation': 3}`

## Schemas & API Contracts (Classes)

### Class `ModelRouter`
> Intelligent multi-stage model router for Jarvis.

**Methods:**
- `def __init__(self, config: Any=None) -> None`
- `def set_telemetry(self, telemetry: Any) -> None`
  - *Connect execution telemetry for adaptive routing feedback.*
- @property
- `def registry(self) -> ModelRegistry`
- @property
- `def strategy(self) -> str`
- `def route(self, task_type: str) -> str`
  - *Determines the target tier for a task and returns the best available model.*
- `def escalate(self, current_model: str) -> str`
  - *Upgrades the model to a higher tier if possible.*
- `def pick_model(self, task_type: str) -> str`
  - *Public entry point — returns the best model name for a task type.*
- `def get_best_available(self, task_type: str) -> str`
- `def route_adaptive(self, classification: dict[str, Any]) -> RoutingDecision`
  - *Cost-optimising model selection using classification signals.*
- `def should_escalate(self, model: str, task_type: str, response: str) -> bool`
  - *Heuristic check if a response quality is too low and needs retry.*
- `def _pick_model_from_tier(self, tier: int) -> str`
- `def is_available(self, model_name: str) -> bool`
- `def list_available(self) -> list[str]`
- `def refresh_available_models(self, base_url: str | None=None, *, force: bool=False, timeout_s: float=3.0) -> list[str]`
- `def list_available_without_refresh(self) -> list[str]`
- `def set_available_models(self, models: Iterable[str]) -> None`
- `def _cfg(self, key: str, default: str) -> str`
- `def _ollama_base_url(self) -> str`
- `def _resolve_available_variant(self, model_name: str) -> str | None`

