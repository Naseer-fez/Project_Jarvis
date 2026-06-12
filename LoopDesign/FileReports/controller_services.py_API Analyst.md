# API Analyst Report: controller\services.py

## Dependencies
- `from __future__ import annotations`
- `import configparser`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Any`
- `from core.agent.agent_loop import AgentLoopEngine`
- `from core.state_machine import StateMachine`
- `from core.autonomy.scheduler import Scheduler`
- `from core.autonomy.autonomy_governor import AutonomyGovernor`
- `from core.autonomy.goal_manager import GoalManager`
- `from core.autonomy.risk_evaluator import RiskEvaluator`
- `from core.desktop.actions import DesktopActionExecutor`
- `from core.desktop.observation import DesktopObserver`
- `from core.llm.client import LLMClientV2`
- `from core.llm.model_router import ModelRouter`
- `from core.llm.telemetry import RoutingTelemetry`
- `from core.planner.planner import TaskPlanner`
- `from core.llm.defaults import DEFAULT_MODEL`
- `from core.memory.hybrid_memory import HybridMemory`
- `from core.profile import UserProfileEngine`
- `from core.proactive.background_monitor import BackgroundMonitor`
- `from core.proactive.notifier import NotificationManager`
- `from core.synthesis import ProfileSynthesizer`
- `from core.runtime.paths import _resolve_path`
- `from core.tools.builtin_tools import register_all_tools`
- `from core.registry.registry import CapabilityRegistry`
- `from core.runtime.event_bus import EventBus`

## Schemas & API Contracts (Classes)

### Class `ControllerSettings`
**Fields/Schema:**
  - `db_path: str`
  - `chroma_path: str`
  - `model_name: str`
  - `embedding_model: str`
  - `base_url: str`
  - `enable_context_titles: bool`
  - `goal_check_interval_seconds: int`
  - `goals_file: Path`



### Class `ControllerServices`
**Fields/Schema:**
  - `memory: HybridMemory`
  - `model_router: ModelRouter`
  - `profile: UserProfileEngine`
  - `llm: LLMClientV2`
  - `synthesizer: ProfileSynthesizer`
  - `state_machine: StateMachine`
  - `task_planner: TaskPlanner`
  - `tool_router: CapabilityRegistry`
  - `risk_evaluator: RiskEvaluator`
  - `autonomy_governor: AutonomyGovernor`
  - `agent_loop: AgentLoopEngine`
  - `goal_manager: GoalManager`
  - `scheduler: Scheduler`
  - `notifier: NotificationManager`
  - `monitor: BackgroundMonitor`
  - `desktop_executor: DesktopActionExecutor`
  - `desktop_observer: DesktopObserver`
  - `desktop_bridge: Any`
  - `event_bus: EventBus`
  - `container: Any`



## Functions & Endpoints

### `build_controller_services`
`def build_controller_services(config: configparser.ConfigParser, *, container: Any=None, db_path: str='memory/memory.db', chroma_path: str='data/chroma', model_name: str=DEFAULT_MODEL, embedding_model: str='all-MiniLM-L6-v2', memory_cls=HybridMemory, model_router_cls=ModelRouter, profile_cls=UserProfileEngine, llm_cls=LLMClientV2, synthesizer_cls=ProfileSynthesizer, state_machine_cls=StateMachine, task_planner_cls=TaskPlanner, tool_router_cls=CapabilityRegistry, risk_evaluator_cls=RiskEvaluator, autonomy_governor_cls=AutonomyGovernor, agent_loop_cls=AgentLoopEngine, goal_manager_cls=GoalManager, scheduler_cls=Scheduler, notifier_cls=NotificationManager, monitor_cls=BackgroundMonitor, register_tools=register_all_tools) -> tuple[ControllerSettings, ControllerServices]`