# Analysis Report for services.py

## Dependencies
- __future__.annotations
- configparser
- dataclasses.dataclass
- pathlib.Path
- typing.Any
- core.agent.agent_loop.AgentLoopEngine
- core.state_machine.StateMachine
- core.autonomy.scheduler.Scheduler
- core.autonomy.autonomy_governor.AutonomyGovernor
- core.autonomy.goal_manager.GoalManager
- core.autonomy.risk_evaluator.RiskEvaluator
- core.desktop.actions.DesktopActionExecutor
- core.desktop.observation.DesktopObserver
- core.llm.client.LLMClientV2
- core.llm.model_router.ModelRouter
- core.llm.telemetry.RoutingTelemetry
- core.planner.planner.TaskPlanner
- core.llm.defaults.DEFAULT_MODEL
- core.memory.hybrid_memory.HybridMemory
- core.profile.UserProfileEngine
- core.proactive.background_monitor.BackgroundMonitor
- core.proactive.notifier.NotificationManager
- core.synthesis.ProfileSynthesizer
- core.runtime.paths._resolve_path
- core.tools.builtin_tools.register_all_tools
- core.registry.registry.CapabilityRegistry
- core.runtime.event_bus.EventBus

## Schemas
- ControllerSettings
- ControllerSettings attribute: db_path
- ControllerSettings attribute: chroma_path
- ControllerSettings attribute: model_name
- ControllerSettings attribute: embedding_model
- ControllerSettings attribute: base_url
- ControllerSettings attribute: enable_context_titles
- ControllerSettings attribute: goal_check_interval_seconds
- ControllerSettings attribute: goals_file
- ControllerServices
- ControllerServices attribute: memory
- ControllerServices attribute: model_router
- ControllerServices attribute: profile
- ControllerServices attribute: llm
- ControllerServices attribute: synthesizer
- ControllerServices attribute: state_machine
- ControllerServices attribute: task_planner
- ControllerServices attribute: tool_router
- ControllerServices attribute: risk_evaluator
- ControllerServices attribute: autonomy_governor
- ControllerServices attribute: agent_loop
- ControllerServices attribute: goal_manager
- ControllerServices attribute: scheduler
- ControllerServices attribute: notifier
- ControllerServices attribute: monitor
- ControllerServices attribute: desktop_executor
- ControllerServices attribute: desktop_observer
- ControllerServices attribute: desktop_bridge
- ControllerServices attribute: event_bus
- ControllerServices attribute: container

## API Contracts
- build_controller_services(config)

## Configuration Variables
None

## Assumptions & Notes
None

