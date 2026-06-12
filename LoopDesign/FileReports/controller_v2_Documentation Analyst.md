# Analysis Report for controller_v2.py

## Dependencies
- __future__.annotations
- asyncio
- configparser
- logging
- uuid
- typing.Any
- core.base_controller.BaseController
- core.controller.intents.handle_goal_intent
- core.controller.intents.handle_preference_intent
- core.controller.intent_router.IntentRouter
- core.controller.services.build_controller_services
- core.llm.defaults.DEFAULT_MODEL
- core.controller.llm_dispatcher.LLMDispatcher
- core.controller.goal_runner.GoalRunner
- core.controller.llm_orchestrator.LLMOrchestrator
- core.controller.memory_subsystem.MemorySubsystem
- core.controller.automation_manager.AutomationManager

## Schemas
- JarvisControllerV2

## API Contracts
- JarvisControllerV2.__init__(self, config, voice, db_path, chroma_path, model_name, embedding_model, container, services, settings)
- JarvisControllerV2._looks_like_desktop_control_request(self, lowered)
- JarvisControllerV2._desktop_control_disabled_message(self)
- JarvisControllerV2._app_launch_disabled_message(self)
- JarvisControllerV2._setup_intent_routes(self)
- JarvisControllerV2.session_summary(self)
- JarvisControllerV2._dashboard_update(self)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes.

