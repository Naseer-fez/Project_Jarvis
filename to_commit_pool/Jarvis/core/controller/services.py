from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from core.agent.agent_loop import AgentLoopEngine
from core.agent.desktop_bridge import DesktopBridge
from core.state_machine import StateMachine
from core.agentic.scheduler import Scheduler
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.goal_manager import GoalManager
from core.autonomy.risk_evaluator import RiskEvaluator
from core.desktop.actions import DesktopActionExecutor
from core.desktop.observation import DesktopObserver
from core.llm.client import LLMClientV2
from core.llm.model_router import ModelRouter
from core.llm.task_planner import TaskPlanner
from core.llm.defaults import DEFAULT_MODEL
from core.memory.hybrid_memory import HybridMemory
from core.profile import UserProfileEngine
from core.proactive.background_monitor import BackgroundMonitor
from core.proactive.notifier import NotificationManager
from core.synthesis import ProfileSynthesizer
from core.runtime.bootstrap import _resolve_path
from core.tools.builtin_tools import register_all_tools
from core.tools.tool_router import ToolRouter
from core.runtime.event_bus import EventBus


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
    desktop_executor: DesktopActionExecutor = None  # type: ignore[assignment]
    desktop_observer: DesktopObserver = None  # type: ignore[assignment]
    desktop_bridge: DesktopBridge = None  # type: ignore[assignment]
    event_bus: EventBus = None  # type: ignore[assignment]
    container: Any = None  # type: ignore[assignment]


def build_controller_services(
    config: configparser.ConfigParser,
    *,
    container: Any = None,
    db_path: str = "memory/memory.db",
    chroma_path: str = "data/chroma",
    model_name: str = DEFAULT_MODEL,
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
    from core.config import JarvisConfig
    if not isinstance(config, JarvisConfig):
        jc = JarvisConfig()
        jc.read_dict(config)
        config = jc

    resolved_db_path = str(
        _resolve_path(
            config.get_str(
                "memory",
                "db_path",
                fallback=config.get_str("memory", "sqlite_file", fallback=db_path),
            )
        )
    )
    resolved_chroma_path = str(
        _resolve_path(
            config.get_str(
                "memory",
                "chroma_path",
                fallback=config.get_str("memory", "chroma_dir", fallback=chroma_path),
            )
        )
    )
    resolved_model_name = config.get_str(
        "models",
        "chat_model",
        fallback=config.get_str(
            "llm",
            "model",
            fallback=config.get_str("ollama", "planner_model", fallback=model_name),
        ),
    )
    resolved_embedding_model = config.get_str(
        "memory",
        "embedding_model",
        fallback=embedding_model,
    )
    base_url = config.get_str("ollama", "base_url", fallback="http://localhost:11434")
    enable_context_titles = config.get_bool(
        "memory",
        "llm_context_titles",
        fallback=True,
    )
    goal_check_interval_seconds = max(
        1,
        config.get_int("proactive", "goal_check_interval_minutes", fallback=5),
    ) * 60
    goals_file = _resolve_path(
        config.get_str("memory", "goals_file", fallback="memory/goals.json")
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

    if container is None:
        from core.runtime.container import ServiceContainer
        container = ServiceContainer()

    # 0. Register EventBus
    if not container.has("event_bus"):
        container.register("event_bus", lambda: EventBus())

    # 1. Register Memory
    if not container.has("memory"):
        container.register(
            "memory",
            lambda: memory_cls(
                settings.db_path,
                chroma_path=settings.chroma_path,
                model_name=settings.embedding_model,
            )
        )

    # 2. Register Model Router
    if not container.has("model_router"):
        container.register("model_router", lambda: model_router_cls(config=config))

    # 3. Register User Profile
    if not container.has("profile"):
        container.register("profile", lambda: profile_cls())

    # 4. Register LLM Client
    if not container.has("llm"):
        container.register(
            "llm",
            lambda: llm_cls(
                hybrid_memory=container.resolve("memory"),
                model=container.resolve("model_router").route("chat"),
                profile=container.resolve("profile"),
                base_url=settings.base_url,
            )
        )

    # 5. Register Profile Synthesizer
    if not container.has("synthesizer"):
        container.register("synthesizer", lambda: synthesizer_cls(container.resolve("llm")))

    # 6. Register State Machine
    if not container.has("state_machine"):
        def make_state_machine():
            import inspect
            sig = inspect.signature(state_machine_cls)
            if "event_bus" in sig.parameters:
                return state_machine_cls(event_bus=container.resolve("event_bus"))
            else:
                inst = state_machine_cls()
                try:
                    inst.event_bus = container.resolve("event_bus")
                except AttributeError:
                    pass
                return inst
        container.register("state_machine", make_state_machine)

    # 7. Register Task Planner
    if not container.has("task_planner"):
        def make_planner():
            try:
                return task_planner_cls(config, llm=container.resolve("llm"))
            except TypeError:
                return task_planner_cls(config)
        container.register("task_planner", make_planner)

    # 8. Register Tool Router
    if not container.has("tool_router"):
        container.register("tool_router", lambda: tool_router_cls())

    # 9. Register Risk Evaluator
    if not container.has("risk_evaluator"):
        container.register("risk_evaluator", lambda: risk_evaluator_cls(config))

    # 10. Register Autonomy Governor
    if not container.has("autonomy_governor"):
        container.register("autonomy_governor", lambda: autonomy_governor_cls(level=3))

    # 11. Register Desktop Executor
    if not container.has("desktop_executor"):
        container.register("desktop_executor", lambda: DesktopActionExecutor(risk_evaluator=container.resolve("risk_evaluator")))

    # 12. Register Desktop Observer
    if not container.has("desktop_observer"):
        container.register("desktop_observer", lambda: DesktopObserver())

    # 13. Register Desktop Bridge
    if not container.has("desktop_bridge"):
        container.register(
            "desktop_bridge",
            lambda: DesktopBridge(container=container)
        )

    # 14. Register Agent Loop Engine
    if not container.has("agent_loop"):
        container.register(
            "agent_loop",
            lambda: agent_loop_cls(
                model=settings.model_name,
                ollama_url=settings.base_url,
                container=container,
            )
        )

    # 15. Register Goal Manager
    if not container.has("goal_manager"):
        container.register("goal_manager", lambda: goal_manager_cls())

    # 16. Register Scheduler
    if not container.has("scheduler"):
        container.register("scheduler", lambda: scheduler_cls())

    # 17. Register Notifier
    if not container.has("notifier"):
        container.register("notifier", lambda: notifier_cls())

    # 18. Register Background Monitor
    if not container.has("monitor"):
        container.register("monitor", lambda: monitor_cls(container.resolve("notifier"), config))

    # Resolve instances and wire them up
    memory = container.resolve(
        "memory",
        db_path=settings.db_path,
        chroma_path=settings.chroma_path,
        model_name=settings.embedding_model,
    )
    model_router = container.resolve("model_router", config=config)
    try:
        model_router.refresh_available_models(base_url=settings.base_url)
    except Exception:
        pass

    profile = container.resolve("profile")
    llm = container.resolve(
        "llm",
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

    synthesizer = container.resolve("synthesizer", llm=llm)
    state_machine = container.resolve("state_machine")
    task_planner = container.resolve("task_planner", config=config, llm=llm)
    tool_router = container.resolve("tool_router")

    register_tools(
        tool_router,
        llm=llm,
        config=config,
    )

    # Dynamic plugin tool loading
    try:
        plugins_dir = config.get("plugins", "directory", fallback="core/plugins")
        resolved_plugins_dir = _resolve_path(plugins_dir)
        if hasattr(tool_router, "load_plugins"):
            loaded_plugins = tool_router.load_plugins(resolved_plugins_dir)
            if loaded_plugins:
                from core.tools.builtin_tools import logger as tools_logger
                tools_logger.info("Loaded dynamic plugins: %s", loaded_plugins)
    except Exception as e:
        from core.tools.builtin_tools import logger as tools_logger
        tools_logger.warning("Failed to load dynamic plugins: %s", e)

    risk_evaluator = container.resolve("risk_evaluator", config=config)
    autonomy_governor = container.resolve("autonomy_governor", level=3)
    desktop_executor = container.resolve("desktop_executor", risk_evaluator=risk_evaluator)
    desktop_observer = container.resolve("desktop_observer")
    desktop_bridge = container.resolve(
        "desktop_bridge",
        container=container,
    )
    agent_loop = container.resolve(
        "agent_loop",
        model=settings.model_name,
        ollama_url=settings.base_url,
        container=container,
    )
    goal_manager = container.resolve("goal_manager")
    scheduler = container.resolve("scheduler")
    notifier = container.resolve("notifier")
    monitor = container.resolve("monitor", notifier=notifier, config=config)
    event_bus = container.resolve("event_bus")

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
        desktop_executor=desktop_executor,
        desktop_observer=desktop_observer,
        desktop_bridge=desktop_bridge,
        event_bus=event_bus,
        container=container,
    )


__all__ = ["ControllerServices", "ControllerSettings", "build_controller_services"]
