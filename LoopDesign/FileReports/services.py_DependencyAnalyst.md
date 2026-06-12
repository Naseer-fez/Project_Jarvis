# Dependency Analysis Report for controller\services.py

## Library Requirements
- from __future__ import annotations
- from core.agent.agent_loop import AgentLoopEngine
- from core.autonomy.autonomy_governor import AutonomyGovernor
- from core.autonomy.goal_manager import GoalManager
- from core.autonomy.risk_evaluator import RiskEvaluator
- from core.autonomy.scheduler import Scheduler
- from core.config import JarvisConfig
- from core.context.context import TaskExecutionContext
- from core.desktop.actions import DesktopActionExecutor
- from core.desktop.observation import DesktopObserver
- from core.executor.engine import DAGExecutor
- from core.llm.client import LLMClientV2
- from core.llm.defaults import DEFAULT_MODEL
- from core.llm.model_router import ModelRouter
- from core.llm.telemetry import RoutingTelemetry
- from core.memory.hybrid_memory import HybridMemory
- from core.planner.planner import TaskPlanner
- from core.proactive.background_monitor import BackgroundMonitor
- from core.proactive.notifier import NotificationManager
- from core.profile import UserProfileEngine
- from core.registry.registry import CapabilityRegistry
- from core.runtime.container import ServiceContainer
- from core.runtime.event_bus import EventBus
- from core.runtime.paths import _resolve_path
- from core.state_machine import StateMachine
- from core.synthesis import ProfileSynthesizer
- from core.tools.builtin_tools import logger
- from core.tools.builtin_tools import register_all_tools
- from dataclasses import dataclass
- from pathlib import Path
- from typing import Any
- import configparser
- import inspect
- import logging

## Service Dependencies
- URL: http://localhost:11434

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
