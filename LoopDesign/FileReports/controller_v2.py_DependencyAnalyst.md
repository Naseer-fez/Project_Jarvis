# Dependency Analysis Report for controller_v2.py

## Library Requirements
- from __future__ import annotations
- from core.base_controller import BaseController
- from core.controller.automation_manager import AutomationManager
- from core.controller.complexity_scorer import classify_request
- from core.controller.goal_runner import GoalRunner
- from core.controller.intent_handlers import register_intent_routes
- from core.controller.intent_router import IntentRouter
- from core.controller.intents import handle_goal_intent
- from core.controller.intents import handle_preference_intent
- from core.controller.llm_dispatcher import LLMDispatcher
- from core.controller.llm_orchestrator import LLMOrchestrator
- from core.controller.memory_subsystem import MemorySubsystem
- from core.controller.services import build_controller_services
- from core.llm.defaults import DEFAULT_MODEL
- from core.voice.voice_layer import VoiceLayer
- from typing import Any
- import asyncio
- import configparser
- import logging
- import uuid

## Service Dependencies
- asyncio.Lock
- asyncio.create_task
- asyncio.get_running_loop

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
