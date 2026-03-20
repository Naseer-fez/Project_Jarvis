from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from core.agent.agent_loop import AgentLoopEngine
from core.agent.state_machine import StateMachine
from core.agentic.scheduler import Scheduler
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.goal_manager import GoalManager
from core.autonomy.risk_evaluator import RiskEvaluator
from core.llm.client import LLMClientV2
from core.llm.model_router import ModelRouter
from core.llm.task_planner import TaskPlanner
from core.memory.hybrid_memory import HybridMemory
from core.profile import UserProfileEngine
from core.proactive.background_monitor import BackgroundMonitor
from core.proactive.notifier import NotificationManager
from core.synthesis import ProfileSynthesizer
from core.tools.builtin_tools import register_all_tools
from core.tools.tool_router import ToolRouter


@dataclass(frozen=True)
class ControllerSettings:
    db_path: str
    chroma_path: str
    model_name: str
    embedding_model: str
    base_url: str
    enable_context_titles: bool
    goal_check_interval_seconds: int
    goals_file: Path


@dataclass(frozen=True)
class ControllerServices:
    memory: HybridMemory
    model_router: ModelRouter
    profile: UserProfileEngine
    llm: LLMClientV2
    synthesizer: ProfileSynthesizer
    state_machine: StateMachine
    task_planner: TaskPlanner
    tool_router: ToolRouter
    risk_evaluator: RiskEvaluator
    autonomy_governor: AutonomyGovernor
    agent_loop: AgentLoopEngine
    goal_manager: GoalManager
    scheduler: Scheduler
    notifier: NotificationManager
    monitor: BackgroundMonitor


def build_controller_services(
    config: configparser.ConfigParser,
    *,
    db_path: str = "memory/memory.db",
    chroma_path: str = "data/chroma",
    model_name: str = "deepseek-r1:8b",
    embedding_model: str = "all-MiniLM-L6-v2",
    memory_cls=HybridMemory,
    model_router_cls=ModelRouter,
    profile_cls=UserProfileEngine,
    llm_cls=LLMClientV2,
    synthesizer_cls=ProfileSynthesizer,
    state_machine_cls=StateMachine,
    task_planner_cls=TaskPlanner,
    tool_router_cls=ToolRouter,
    risk_evaluator_cls=RiskEvaluator,
    autonomy_governor_cls=AutonomyGovernor,
    agent_loop_cls=AgentLoopEngine,
    goal_manager_cls=GoalManager,
    scheduler_cls=Scheduler,
    notifier_cls=NotificationManager,
    monitor_cls=BackgroundMonitor,
    register_tools=register_all_tools,
) -> tuple[ControllerSettings, ControllerServices]:
    resolved_db_path = config.get(
        "memory",
        "db_path",
        fallback=config.get("memory", "sqlite_file", fallback=db_path),
    )
    resolved_chroma_path = config.get(
        "memory",
        "chroma_path",
        fallback=config.get("memory", "chroma_dir", fallback=chroma_path),
    )
    resolved_model_name = config.get(
        "models",
        "chat_model",
        fallback=config.get(
            "llm",
            "model",
            fallback=config.get("ollama", "planner_model", fallback=model_name),
        ),
    )
    resolved_embedding_model = config.get(
        "memory",
        "embedding_model",
        fallback=embedding_model,
    )
    base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
    enable_context_titles = config.getboolean(
        "memory",
        "llm_context_titles",
        fallback=True,
    )
    goal_check_interval_seconds = max(
        1,
        config.getint("proactive", "goal_check_interval_minutes", fallback=5),
    ) * 60
    goals_file = Path(
        config.get("memory", "goals_file", fallback="memory/goals.json")
    )

    settings = ControllerSettings(
        db_path=resolved_db_path,
        chroma_path=resolved_chroma_path,
        model_name=resolved_model_name,
        embedding_model=resolved_embedding_model,
        base_url=base_url,
        enable_context_titles=enable_context_titles,
        goal_check_interval_seconds=goal_check_interval_seconds,
        goals_file=goals_file,
    )

    memory = memory_cls(
        settings.db_path,
        chroma_path=settings.chroma_path,
        model_name=settings.embedding_model,
    )
    model_router = model_router_cls(config=config)
    profile = profile_cls()
    llm = llm_cls(
        hybrid_memory=memory,
        model=model_router.route("chat"),
        profile=profile,
        base_url=settings.base_url,
    )
    llm.set_router(model_router)
    if hasattr(memory, "set_llm"):
        memory.set_llm(
            llm,
            enable_context_titles=settings.enable_context_titles,
        )
    try:
        setattr(llm, "profile", profile)
    except Exception:
        pass

    synthesizer = synthesizer_cls(llm)
    state_machine = state_machine_cls()
    task_planner = task_planner_cls(config)
    tool_router = tool_router_cls()
    register_tools(
        tool_router,
        llm=llm,
        config=config,
    )
    risk_evaluator = risk_evaluator_cls(config)
    autonomy_governor = autonomy_governor_cls(level=3)
    agent_loop = agent_loop_cls(
        state_machine=state_machine,
        task_planner=task_planner,
        tool_router=tool_router,
        risk_evaluator=risk_evaluator,
        autonomy_governor=autonomy_governor,
        model=settings.model_name,
        ollama_url=settings.base_url,
        llm=llm,
    )
    goal_manager = goal_manager_cls()
    scheduler = scheduler_cls()
    notifier = notifier_cls()
    monitor = monitor_cls(notifier, config)

    return settings, ControllerServices(
        memory=memory,
        model_router=model_router,
        profile=profile,
        llm=llm,
        synthesizer=synthesizer,
        state_machine=state_machine,
        task_planner=task_planner,
        tool_router=tool_router,
        risk_evaluator=risk_evaluator,
        autonomy_governor=autonomy_governor,
        agent_loop=agent_loop,
        goal_manager=goal_manager,
        scheduler=scheduler,
        notifier=notifier,
        monitor=monitor,
    )


__all__ = ["ControllerServices", "ControllerSettings", "build_controller_services"]
