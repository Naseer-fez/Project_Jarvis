# Dependency Analysis Report for agent\agent_loop.py

## Library Requirements
- from __future__ import annotations
- from core.autonomy.autonomy_governor import AutonomyGovernor
- from core.autonomy.autonomy_governor import AutonomyLevel
- from core.autonomy.risk_evaluator import RiskEvaluator
- from core.context.context import TaskExecutionContext
- from core.metrics.confidence import ConfidenceModel
- from core.planner.planner import TaskPlanner
- from core.registry.registry import CapabilityRegistry
- from core.registry.registry import ToolObservation
- from core.state_machine import State
- from core.state_machine import StateMachine
- from dataclasses import dataclass
- from dataclasses import field
- from typing import Any
- from typing import Optional
- import asyncio
- import httpx
- import inspect
- import logging
- import re
- import sys
- import time
- import traceback

## Service Dependencies
- URL: http://localhost:11434
- asyncio.Event
- asyncio.Lock
- asyncio.timeout
- asyncio.to_thread
- httpx.AsyncClient

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
