# Core Directory Manifest

**Folder**: d:\AI\Jarvis\core

**High-Level Purpose**: The core folder is the brain and primary engine of the Jarvis AI system. It encompasses the entirety of the execution pipeline, including LLM orchestration, semantic and episodic memory retrieval, external tool interfacing, desktop automation, autonomous goal management, hardware integrations, and system-level introspection. It orchestrates the flow of data from inputs (voice, text, proactive triggers) through the LLM planning layers, and dispatches actions to local or external environments.

## Root Level Files
- __init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- base_controller.py - Required Roles: Runtime Investigator
- controller_v2.py - Required Roles: Runtime Investigator
- permission_matrix.py - Required Roles: Runtime Investigator
- profile.py - Required Roles: Runtime Investigator
- state_machine.py - Required Roles: Runtime Investigator
- synthesis.py - Required Roles: Prompt Recovery Specialist, Runtime Investigator

## Subdirectory: agent
- **Purpose**: Components and logic related to agent subsystem.
- agent\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- agent\agent_loop.py - Required Roles: Runtime Investigator

## Subdirectory: agentic
- **Purpose**: Components and logic related to agentic subsystem.
- agentic\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- agentic\goal_manager.py - Required Roles: Runtime Investigator

## Subdirectory: automation
- **Purpose**: Components and logic related to automation subsystem.
- automation\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- automation\live_automation.py - Required Roles: API Analyst, Runtime Investigator
- automation\payload_extractor.py - Required Roles: API Analyst, Runtime Investigator
- automation\rag_ingester.py - Required Roles: API Analyst, Runtime Investigator
- automation\scan_pipeline.py - Required Roles: API Analyst, Runtime Investigator
- automation\scan_rules.py - Required Roles: API Analyst, Runtime Investigator

## Subdirectory: autonomy
- **Purpose**: Components and logic related to autonomy subsystem.
- autonomy\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- autonomy\autonomy_governor.py - Required Roles: Runtime Investigator
- autonomy\goal_manager.py - Required Roles: Runtime Investigator
- autonomy\risk_evaluator.py - Required Roles: Runtime Investigator
- autonomy\scheduler.py - Required Roles: Runtime Investigator

## Subdirectory: capability
- **Purpose**: Components and logic related to capability subsystem.
- capability\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- capability\base.py - Required Roles: API Analyst, Runtime Investigator

## Subdirectory: config
- **Purpose**: Components and logic related to config subsystem.
- config\__init__.py - Required Roles: Dependency Analyst, Configuration Analyst, Runtime Investigator
- config\defaults.py - Required Roles: Configuration Analyst, Runtime Investigator

## Subdirectory: context
- **Purpose**: Components and logic related to context subsystem.
- context\__init__.py - Required Roles: Data Model Analyst, Dependency Analyst, Runtime Investigator
- context\context.py - Required Roles: Data Model Analyst, Runtime Investigator

## Subdirectory: controller
- **Purpose**: Components and logic related to controller subsystem.
- controller\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- controller\automation_manager.py - Required Roles: API Analyst, Runtime Investigator
- controller\complexity_scorer.py - Required Roles: API Analyst, Runtime Investigator
- controller\goal_runner.py - Required Roles: API Analyst, Runtime Investigator
- controller\intent_handlers.py - Required Roles: API Analyst, Runtime Investigator
- controller\intent_router.py - Required Roles: API Analyst, Runtime Investigator
- controller\intents.py - Required Roles: API Analyst, Runtime Investigator
- controller\llm_dispatcher.py - Required Roles: API Analyst, Runtime Investigator
- controller\llm_orchestrator.py - Required Roles: API Analyst, Runtime Investigator
- controller\memory_subsystem.py - Required Roles: API Analyst, Runtime Investigator
- controller\request_rules.py - Required Roles: API Analyst, Runtime Investigator
- controller\services.py - Required Roles: API Analyst, Runtime Investigator
- controller\web_search.py - Required Roles: API Analyst, Runtime Investigator

## Subdirectory: desktop
- **Purpose**: Components and logic related to desktop subsystem.
- desktop\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- desktop\actions.py - Required Roles: API Analyst, Runtime Investigator
- desktop\contracts.py - Required Roles: API Analyst, Runtime Investigator
- desktop\mission.py - Required Roles: API Analyst, Runtime Investigator
- desktop\observation.py - Required Roles: API Analyst, Runtime Investigator
- desktop\shortcuts.py - Required Roles: API Analyst, Runtime Investigator

## Subdirectory: execution
- **Purpose**: Components and logic related to execution subsystem.
- execution\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- execution\dispatcher.py - Required Roles: Runtime Investigator

## Subdirectory: executor
- **Purpose**: Components and logic related to executor subsystem.
- executor\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- executor\dag.py - Required Roles: Dependency Analyst, Runtime Investigator
- executor\engine.py - Required Roles: Dependency Analyst, Runtime Investigator

## Subdirectory: hardware
- **Purpose**: Components and logic related to hardware subsystem.
- hardware\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- hardware\device_registry.py - Required Roles: API Analyst, Runtime Investigator
- hardware\serial_controller.py - Required Roles: API Analyst, Runtime Investigator

## Subdirectory: introspection
- **Purpose**: Components and logic related to introspection subsystem.
- introspection\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- introspection\health.py - Required Roles: Runtime Investigator

## Subdirectory: llm
- **Purpose**: Components and logic related to llm subsystem.
- llm\__init__.py - Required Roles: Dependency Analyst, Prompt Recovery Specialist, Runtime Investigator
- llm\client.py - Required Roles: API Analyst, Prompt Recovery Specialist, Runtime Investigator
- llm\cloud_client.py - Required Roles: API Analyst, Prompt Recovery Specialist, Runtime Investigator
- llm\defaults.py - Required Roles: Prompt Recovery Specialist, Runtime Investigator
- llm\model_router.py - Required Roles: Data Model Analyst, Prompt Recovery Specialist, Runtime Investigator
- llm\model_spec.py - Required Roles: Data Model Analyst, Prompt Recovery Specialist, Runtime Investigator
- llm\ollama_client.py - Required Roles: API Analyst, Prompt Recovery Specialist, Runtime Investigator
- llm\telemetry.py - Required Roles: Prompt Recovery Specialist, Runtime Investigator

## Subdirectory: logging
- **Purpose**: Components and logic related to logging subsystem.
- logging\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- logging\logger.py - Required Roles: Runtime Investigator

## Subdirectory: memory
- **Purpose**: Components and logic related to memory subsystem.
- memory\__init__.py - Required Roles: Data Model Analyst, Dependency Analyst, Runtime Investigator
- memory\code_indexer.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\code_indexer_service.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\context_compressor.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\embeddings.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\hybrid_memory.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\retriever.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\semantic_memory.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\sqlite_pool.py - Required Roles: Data Model Analyst, Runtime Investigator
- memory\sqlite_storage.py - Required Roles: Data Model Analyst, Runtime Investigator

## Subdirectory: metrics
- **Purpose**: Components and logic related to metrics subsystem.
- metrics\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- metrics\confidence.py - Required Roles: Runtime Investigator

## Subdirectory: ops
- **Purpose**: Components and logic related to ops subsystem.
- ops\__init__.py - Required Roles: Dependency Analyst, Configuration Analyst, Runtime Investigator
- ops\production.py - Required Roles: Configuration Analyst, Runtime Investigator

## Subdirectory: planner
- **Purpose**: Components and logic related to planner subsystem.
- planner\__init__.py - Required Roles: Dependency Analyst, Prompt Recovery Specialist, Runtime Investigator
- planner\planner.py - Required Roles: Prompt Recovery Specialist, Runtime Investigator

## Subdirectory: plugins
- **Purpose**: Components and logic related to plugins subsystem.
- plugins\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator

## Subdirectory: proactive
- **Purpose**: Components and logic related to proactive subsystem.
- proactive\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- proactive\background_monitor.py - Required Roles: Runtime Investigator
- proactive\notifier.py - Required Roles: Runtime Investigator

## Subdirectory: registry
- **Purpose**: Components and logic related to registry subsystem.
- registry\__init__.py - Required Roles: Dependency Analyst, Configuration Analyst, Runtime Investigator
- registry\base.py - Required Roles: Dependency Analyst, Configuration Analyst, Runtime Investigator
- registry\registry.py - Required Roles: Dependency Analyst, Configuration Analyst, Runtime Investigator

## Subdirectory: runtime
- **Purpose**: Components and logic related to runtime subsystem.
- runtime\__init__.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\bootstrap.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\container.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\dashboard_runtime.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\entrypoint.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\event_bus.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\import_validator.py - Required Roles: Dependency Analyst, Runtime Investigator
- runtime\paths.py - Required Roles: Dependency Analyst, Runtime Investigator

## Subdirectory: security
- **Purpose**: Components and logic related to security subsystem.
- security\__init__.py - Required Roles: Dependency Analyst, Configuration Analyst, Runtime Investigator
- security\auth.py - Required Roles: Configuration Analyst, Runtime Investigator

## Subdirectory: tools
- **Purpose**: Components and logic related to tools subsystem.
- tools\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- tools\auto_clicker.py - Required Roles: API Analyst, Runtime Investigator
- tools\builtin_tools.py - Required Roles: API Analyst, Runtime Investigator
- tools\fast_search\fast_search.cpp - Required Roles: API Analyst, Runtime Investigator
- tools\fast_search\fast_search.exe - Required Roles: API Analyst, Runtime Investigator
- tools\fast_search_tool.py - Required Roles: API Analyst, Runtime Investigator
- tools\gui_control.py - Required Roles: API Analyst, Runtime Investigator
- tools\hardware_tools.py - Required Roles: API Analyst, Runtime Investigator
- tools\path_utils.py - Required Roles: API Analyst, Runtime Investigator
- tools\screen.py - Required Roles: API Analyst, Runtime Investigator
- tools\system_automation.py - Required Roles: API Analyst, Runtime Investigator
- tools\universal_converter.py - Required Roles: API Analyst, Runtime Investigator
- tools\vision.py - Required Roles: API Analyst, Runtime Investigator
- tools\web_tools.py - Required Roles: API Analyst, Runtime Investigator
- tools\fast_search (Directory) - Required Roles: Dependency Analyst, Runtime Investigator

## Subdirectory: types
- **Purpose**: Components and logic related to types subsystem.
- types\__init__.py - Required Roles: Data Model Analyst, Dependency Analyst, Runtime Investigator
- types\common.py - Required Roles: Data Model Analyst, Runtime Investigator

## Subdirectory: voice
- **Purpose**: Components and logic related to voice subsystem.
- voice\__init__.py - Required Roles: API Analyst, Dependency Analyst, Runtime Investigator
- voice\audio.py - Required Roles: API Analyst, Runtime Investigator
- voice\audio_input.py - Required Roles: API Analyst, Runtime Investigator
- voice\audio_playback.py - Required Roles: API Analyst, Runtime Investigator
- voice\audio_utils.py - Required Roles: API Analyst, Runtime Investigator
- voice\stt.py - Required Roles: API Analyst, Runtime Investigator
- voice\tts.py - Required Roles: API Analyst, Runtime Investigator
- voice\voice_layer.py - Required Roles: API Analyst, Runtime Investigator
- voice\voice_loop.py - Required Roles: API Analyst, Runtime Investigator
- voice\wake_word.py - Required Roles: API Analyst, Runtime Investigator

