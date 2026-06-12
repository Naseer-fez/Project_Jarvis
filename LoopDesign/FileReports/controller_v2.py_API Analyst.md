# API Analyst Report: controller_v2.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import configparser`
- `import logging`
- `import uuid`
- `from typing import Any`
- `from core.base_controller import BaseController`
- `from core.controller.intents import handle_goal_intent`
- `from core.controller.intents import handle_preference_intent`
- `from core.controller.intent_router import IntentRouter`
- `from core.controller.services import build_controller_services`
- `from core.llm.defaults import DEFAULT_MODEL`
- `from core.controller.llm_dispatcher import LLMDispatcher`
- `from core.controller.goal_runner import GoalRunner`
- `from core.controller.llm_orchestrator import LLMOrchestrator`
- `from core.controller.memory_subsystem import MemorySubsystem`
- `from core.controller.automation_manager import AutomationManager`

## Schemas & API Contracts (Classes)

### Class `JarvisControllerV2(BaseController)`
**Methods:**
- `def __init__(self, config: configparser.ConfigParser | None=None, voice: bool=False, db_path: str='memory/memory.db', chroma_path: str='data/chroma', model_name: str=DEFAULT_MODEL, embedding_model: str='all-MiniLM-L6-v2', container: Any=None, services: Any=None, settings: Any=None) -> None`
- `async def initialize(self) -> dict[str, Any]`
- `async def _handle_goal_intent(self, text: str, user_input: str) -> str | None`
- `async def _handle_preference_intent(self, text: str, user_input: str) -> str | None`
- `async def _dispatch_llm(self, text: str, classification: dict, trace_id: str) -> str`
- `def _looks_like_desktop_control_request(self, lowered: str) -> bool`
- `def _desktop_control_disabled_message(self) -> str`
- `def _app_launch_disabled_message(self) -> str`
- `def _setup_intent_routes(self) -> None`
- `async def process(self, user_input: str, trace_id: str | None=None) -> str`
- `def session_summary(self) -> dict[str, Any]`
- `async def startup(self) -> None`
- `async def start(self) -> None`
  - *Alias for startup() to maintain backward compatibility.*
- `async def run_cli(self) -> None`
- `async def shutdown(self) -> None`
- `def _dashboard_update(self, **kwargs: Any) -> None`

