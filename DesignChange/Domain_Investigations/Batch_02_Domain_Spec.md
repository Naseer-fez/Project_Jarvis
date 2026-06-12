# Domain Specification: Batch_02

## Responsibilities
This domain handles the following components:
- **core\base_controller.py**: Encompasses classes BaseController
- **core\controller_v2.py**: Encompasses classes JarvisControllerV2
- **core\permission_matrix.py**: Encompasses classes PermissionResult, PermissionMatrix
- **core\profile.py**: Encompasses classes UserProfileEngine
- **core\state_machine.py**: Encompasses classes IllegalTransitionError, State, StateGuard, StateMachine
- **core\synthesis.py**: Encompasses classes ProfileSynthesizer
- **core\__init__.py**: Encompasses classes None
- **core\agent\agent_loop.py**: Encompasses classes ExecutionTrace, AgentLoopEngine
- **core\agent\__init__.py**: Encompasses classes None
- **core\automation\live_automation.py**: Encompasses classes AutomationStats, LiveAutomationEngine
- **core\automation\payload_extractor.py**: Encompasses classes PayloadExtractor
- **core\automation\rag_ingester.py**: Encompasses classes RagIngester
- **core\automation\scan_pipeline.py**: Encompasses classes ScanBatch, ScanPipeline
- **core\automation\scan_rules.py**: Encompasses classes ScanRoute
- **core\automation\__init__.py**: Encompasses classes None
- **core\autonomy\autonomy_governor.py**: Encompasses classes AutonomyLevel, AutonomyGovernor
- **core\autonomy\goal_manager.py**: Encompasses classes GoalStatus, Goal, GoalManager
- **core\autonomy\risk_evaluator.py**: Encompasses classes RiskLevel, RiskResult, RiskEvaluator
- **core\autonomy\scheduler.py**: Encompasses classes ScheduleStatus, ScheduledMission, Scheduler
- **core\autonomy\__init__.py**: Encompasses classes None
- **core\capability\base.py**: Encompasses classes Capability, ToolObservation
- **core\capability\__init__.py**: Encompasses classes None
- **core\config\defaults.py**: Encompasses classes None
- **core\config\__init__.py**: Encompasses classes JarvisConfig
- **core\context\context.py**: Encompasses classes TaskExecutionContext
- **core\context\__init__.py**: Encompasses classes None
- **core\controller\automation_manager.py**: Encompasses classes AutomationManager
- **core\controller\complexity_scorer.py**: Encompasses classes None
- **core\controller\goal_runner.py**: Encompasses classes GoalRunner
- **core\controller\intents.py**: Encompasses classes GoalIntentResult
- **core\controller\intent_handlers.py**: Encompasses classes None
- **core\controller\intent_router.py**: Encompasses classes IntentRoute, IntentRouter
- **core\controller\llm_dispatcher.py**: Encompasses classes LLMDispatcher
- **core\controller\llm_orchestrator.py**: Encompasses classes LLMOrchestrator
- **core\controller\memory_subsystem.py**: Encompasses classes MemorySubsystem
- **core\controller\request_rules.py**: Encompasses classes None
- **core\controller\services.py**: Encompasses classes ControllerSettings, ControllerServices
- **core\controller\web_search.py**: Encompasses classes None
- **core\controller\__init__.py**: Encompasses classes None
- **core\desktop\actions.py**: Encompasses classes DesktopActionExecutor
- **core\desktop\contracts.py**: Encompasses classes DesktopActionType, DesktopRiskTier, DesktopActionStatus, DesktopAction, DesktopActionResult, ScreenTarget, DesktopObservation, DesktopChange, ApprovalDecision
- **core\desktop\mission.py**: Encompasses classes DesktopMissionStatus, RecoveryDecision, MissionStepRecord, MissionExecutionRecord, DesktopMissionExecutor
- **core\desktop\observation.py**: Encompasses classes DesktopObserver
- **core\desktop\shortcuts.py**: Encompasses classes DesktopCommandPlan
- **core\desktop\__init__.py**: Encompasses classes None
- **core\executor\dag.py**: Encompasses classes DependencyGraphError, PlanDAG
- **core\executor\engine.py**: Encompasses classes DAGExecutor
- **core\executor\__init__.py**: Encompasses classes None
- **core\hardware\device_registry.py**: Encompasses classes HardwareDevice, DeviceRegistry
- **core\hardware\serial_controller.py**: Encompasses classes SerialController
- **core\hardware\__init__.py**: Encompasses classes None
- **core\introspection\health.py**: Encompasses classes HealthStatus, HealthCheck, HealthReport
- **core\introspection\__init__.py**: Encompasses classes None
- **core\llm\client.py**: Encompasses classes LLMClientV2
- **core\llm\cloud_client.py**: Encompasses classes CloudLLMClient
- **core\llm\defaults.py**: Encompasses classes None
- **core\llm\model_router.py**: Encompasses classes ModelRouter
- **core\llm\model_spec.py**: Encompasses classes ModelSpec, RoutingDecision, ModelRegistry
- **core\llm\ollama_client.py**: Encompasses classes OllamaTransientError, OllamaClient
- **core\llm\telemetry.py**: Encompasses classes _CallRecord, ModelStats, RoutingTelemetry
- **core\llm\__init__.py**: Encompasses classes None
- **core\logging\logger.py**: Encompasses classes AuditLog, JSONFormatter, FlushingQueueListener, JarvisQueueHandler
- **core\logging\__init__.py**: Encompasses classes None
- **core\memory\code_indexer.py**: Encompasses classes None
- **core\memory\code_indexer_service.py**: Encompasses classes CodeIndexerService
- **core\memory\context_compressor.py**: Encompasses classes ContextCompressor
- **core\memory\embeddings.py**: Encompasses classes DeterministicMockSentenceTransformer, EmbeddingManager
- **core\memory\hybrid_memory.py**: Encompasses classes HybridMemory, _Fact, _Fact
- **core\memory\retriever.py**: Encompasses classes MemoryRetriever
- **core\memory\semantic_memory.py**: Encompasses classes SemanticMemory
- **core\memory\sqlite_pool.py**: Encompasses classes SQLitePool
- **core\memory\sqlite_storage.py**: Encompasses classes SQLiteStorage
- **core\memory\__init__.py**: Encompasses classes None
- **core\metrics\confidence.py**: Encompasses classes ConfidenceModel
- **core\metrics\__init__.py**: Encompasses classes None
- **core\ops\production.py**: Encompasses classes ProductionCheck
- **core\ops\__init__.py**: Encompasses classes None
- **core\planner\planner.py**: Encompasses classes TaskPlanner
- **core\planner\__init__.py**: Encompasses classes None
- **core\plugins\__init__.py**: Encompasses classes PluginCatalog, PluginManifest, PluginManifestError
- **core\proactive\background_monitor.py**: Encompasses classes BackgroundMonitor
- **core\proactive\notifier.py**: Encompasses classes NotificationManager
- **core\proactive\__init__.py**: Encompasses classes None
- **core\registry\base.py**: Encompasses classes RiskLevel, Capability
- **core\registry\registry.py**: Encompasses classes FunctionCapability, DesktopCapability, CapabilityRegistry
- **core\registry\__init__.py**: Encompasses classes None
- **core\security\auth.py**: Encompasses classes AuthUser, AuthManager
- **core\security\__init__.py**: Encompasses classes None
- **core\tools\auto_clicker.py**: Encompasses classes None
- **core\tools\builtin_tools.py**: Encompasses classes None
- **core\tools\fast_search_tool.py**: Encompasses classes PythonSearchEngine
- **core\tools\gui_control.py**: Encompasses classes None
- **core\tools\hardware_tools.py**: Encompasses classes None
- **core\tools\path_utils.py**: Encompasses classes None
- **core\tools\screen.py**: Encompasses classes None
- **core\tools\system_automation.py**: Encompasses classes ToolResult
- **core\tools\universal_converter.py**: Encompasses classes None
- **core\tools\vision.py**: Encompasses classes VisionTool
- **core\tools\web_tools.py**: Encompasses classes SupportsQuickLLM, SearchSettings, SearchResult, WebToolService
- **core\tools\__init__.py**: Encompasses classes None
- **core\types\common.py**: Encompasses classes ToolResult, IntegrationRiskLevel
- **core\types\__init__.py**: Encompasses classes None
- **core\voice\audio.py**: Encompasses classes None
- **core\voice\audio_input.py**: Encompasses classes AudioInputSerde, AudioInputMixin
- **core\voice\audio_playback.py**: Encompasses classes AudioPlayer
- **core\voice\audio_utils.py**: Encompasses classes None
- **core\voice\stt.py**: Encompasses classes TranscriptResult, STT, SpeechToText
- **core\voice\tts.py**: Encompasses classes TTS, TextToSpeech, _TextToSpeechStub
- **core\voice\voice_layer.py**: Encompasses classes VoiceLayer
- **core\voice\voice_loop.py**: Encompasses classes VoiceLoop
- **core\voice\wake_word.py**: Encompasses classes WakeWordDetector
- **core\voice\__init__.py**: Encompasses classes None

## Internal Structure
### Class: BaseController
- **Methods**: __init__, register_subsystem
### Class: JarvisControllerV2
- **Methods**: __init__, _looks_like_desktop_control_request, _desktop_control_disabled_message, _app_launch_disabled_message, _setup_intent_routes, session_summary, _dashboard_update
### Class: PermissionResult
- **Methods**: has_blocked, needs_confirmation
### Class: PermissionMatrix
- **Methods**: __init__, evaluate, _parse_csv
### Class: UserProfileEngine
- **Methods**: __init__, _fresh_defaults, _load, save, update_from_conversation, apply_delta, get_system_prompt_injection, get_communication_style, interaction_count
### Class: IllegalTransitionError
- **Methods**: 
### Class: State
- **Methods**: 
### Class: StateGuard
- **Methods**: __init__, __enter__, __exit__
### Class: StateMachine
- **Methods**: __init__, state, add_listener, remove_listener, can_transition, get_valid_transitions, get_transition_graph, _notify, transition, reset, force_idle, transition_to, __enter__, __exit__
### Class: ProfileSynthesizer
- **Methods**: __init__, should_run
### Class: ExecutionTrace
- **Methods**: close, to_dict
### Class: AgentLoopEngine
- **Methods**: __init__, request_interrupt, _check_interrupt, _ensure_thinking_state, _normalize_steps, _plan_summary, _fallback_reflection, _stop
### Class: AutomationStats
- **Methods**: 
### Class: LiveAutomationEngine
- **Methods**: __init__, _build_command_scan_batch, _build_ingest_scan_batch, _scan_readiness, _handle_scan_failure, _apply_scan_summary, status, status_line, _extract_text_payload, _file_ready, _extract_command, _read_text_file, _move_to_failed, _relocate, _unique_path, _fingerprint, _remember_file, _remember_fingerprint, _ensure_directories, _append_log, _load_state, _save_state, _extract_metadata_value
### Class: PayloadExtractor
- **Methods**: __init__, extract_text_payload, extract_text_from_image, extract_text_from_video
### Class: RagIngester
- **Methods**: __init__, chunk_text
### Class: ScanBatch
- **Methods**: 
### Class: ScanPipeline
- **Methods**: __init__
### Class: ScanRoute
- **Methods**: 
### Class: AutonomyLevel
- **Methods**: 
### Class: AutonomyGovernor
- **Methods**: __init__, register_read_only_tool, register_write_tool, _is_known_tool, _is_write_tool, can_execute, requires_confirmation, escalate, describe
### Class: GoalStatus
- **Methods**: 
### Class: Goal
- **Methods**: start, complete, fail, cancel, pause, resume, is_terminal, to_dict
### Class: GoalManager
- **Methods**: __init__, create_goal, get_goal, start_goal, complete_goal, fail_goal, cancel_goal, pause_goal, resume_goal, update_goal, remove_goal, next_goal, active_goals, all_goals, get_goals_by_status, get_subgoals, snapshot, restore
### Class: RiskLevel
- **Methods**: label
### Class: RiskResult
- **Methods**: is_blocked, requires_confirmation, summary
### Class: RiskEvaluator
- **Methods**: __init__, register_critical_action, register_confirm_action, register_high_action, register_medium_action, register_low_action, _load_config, evaluate, evaluate_plan
### Class: ScheduleStatus
- **Methods**: 
### Class: ScheduledMission
- **Methods**: is_due, next_retry_delay, mark_completed, mark_cancelled, schedule_retry, to_dict
### Class: Scheduler
- **Methods**: __init__, enqueue, due, get, cancel, pending, snapshot, restore
### Class: Capability
- **Methods**: is_write_operation
### Class: ToolObservation
- **Methods**: to_dict
### Class: JarvisConfig
- **Methods**: get_str, get_bool, get_int
### Class: TaskExecutionContext
- **Methods**: __init__, log, get, set, __getitem__, __setitem__, __contains__, to_dict, __enter__, __exit__
### Class: AutomationManager
- **Methods**: __init__
### Class: GoalRunner
- **Methods**: __init__, load_goal_state, persist_goal_state
### Class: GoalIntentResult
- **Methods**: 
### Class: IntentRoute
- **Methods**: 
### Class: IntentRouter
- **Methods**: __init__, register
### Class: LLMDispatcher
- **Methods**: __init__
### Class: LLMOrchestrator
- **Methods**: __init__
### Class: MemorySubsystem
- **Methods**: __init__, update_profile, _schedule_synthesis
### Class: ControllerSettings
- **Methods**: 
### Class: ControllerServices
- **Methods**: 
### Class: DesktopActionExecutor
- **Methods**: __init__, evaluate_risk, requires_approval, _audit, _contains_sensitive_text, _result, _default_handlers
### Class: DesktopActionType
- **Methods**: 
### Class: DesktopRiskTier
- **Methods**: 
### Class: DesktopActionStatus
- **Methods**: 
### Class: DesktopAction
- **Methods**: action_name, to_dict
### Class: DesktopActionResult
- **Methods**: duration_seconds, to_dict
### Class: ScreenTarget
- **Methods**: to_dict
### Class: DesktopObservation
- **Methods**: to_dict
### Class: DesktopChange
- **Methods**: to_dict
### Class: ApprovalDecision
- **Methods**: to_dict
### Class: DesktopMissionStatus
- **Methods**: 
### Class: RecoveryDecision
- **Methods**: 
### Class: MissionStepRecord
- **Methods**: to_dict
### Class: MissionExecutionRecord
- **Methods**: close, duration_seconds, explain, to_dict
### Class: DesktopMissionExecutor
- **Methods**: __init__, _summary_for, _audit
### Class: DesktopObserver
- **Methods**: __init__, compare, _default_capture_screen, _default_active_window, _default_ocr
### Class: DesktopCommandPlan
- **Methods**: 
### Class: DependencyGraphError
- **Methods**: 
### Class: PlanDAG
- **Methods**: __init__, topological_sort
### Class: DAGExecutor
- **Methods**: __init__
### Class: HardwareDevice
- **Methods**: __init__
### Class: DeviceRegistry
- **Methods**: __init__, _load_from_config, register_device, get_device, list_devices
### Class: SerialController
- **Methods**: __init__, is_connected, connect, send, close
### Class: HealthStatus
- **Methods**: 
### Class: HealthCheck
- **Methods**: 
### Class: HealthReport
- **Methods**: has_failures, is_healthy, ollama_reachable, summary
### Class: LLMClientV2
- **Methods**: __init__, set_router, set_telemetry, _record_telemetry, chat, _messages_to_prompt
### Class: CloudLLMClient
- **Methods**: __init__, _extract_openai_usage
### Class: ModelRouter
- **Methods**: __init__, set_telemetry, registry, strategy, route, escalate, pick_model, get_best_available, route_adaptive, should_escalate, _pick_model_from_tier, is_available, list_available, refresh_available_models, list_available_without_refresh, set_available_models, _cfg, _ollama_base_url, _resolve_available_variant
### Class: ModelSpec
- **Methods**: 
### Class: RoutingDecision
- **Methods**: 
### Class: ModelRegistry
- **Methods**: __init__, get, get_tier, get_weight, all_specs, by_provider, by_tier, get_cheapest_capable, register, _load_config_overrides, _parse_override
### Class: OllamaTransientError
- **Methods**: 
### Class: OllamaClient
- **Methods**: __init__
### Class: _CallRecord
- **Methods**: 
### Class: ModelStats
- **Methods**: 
### Class: RoutingTelemetry
- **Methods**: __init__, record, get_model_stats, get_reliability, get_avg_latency, get_cost_estimate, summary, save_to_file, load_from_file, _resolve_cost_weight
### Class: AuditLog
- **Methods**: __init__, _start_worker, _write_worker, write, stop, verify
### Class: JSONFormatter
- **Methods**: format
### Class: FlushingQueueListener
- **Methods**: handle, flush
### Class: JarvisQueueHandler
- **Methods**: prepare
### Class: CodeIndexerService
- **Methods**: __init__
### Class: ContextCompressor
- **Methods**: __init__, _get_item_text, _compress_preferences, _compress_episodes, _compress_conversations, _clean, _truncate, _estimate_tokens, _deduplicate, explain
### Class: DeterministicMockSentenceTransformer
- **Methods**: __init__, get_sentence_embedding_dimension, _get_word_vector, encode, _encode_single
### Class: EmbeddingManager
- **Methods**: __init__, is_ready, dimension, info, clear_cache
### Class: HybridMemory
- **Methods**: __init__, _track_background_task, set_llm, _query_tokens, _score_text, stats
### Class: _Fact
- **Methods**: __init__, __repr__
### Class: _Fact
- **Methods**: __init__
### Class: MemoryRetriever
- **Methods**: __init__, query_tokens, score_text
### Class: SemanticMemory
- **Methods**: __init__, _collection, is_ready
### Class: SQLitePool
- **Methods**: __init__
### Class: SQLiteStorage
- **Methods**: __init__
### Class: ConfidenceModel
- **Methods**: __init__, update, score
### Class: ProductionCheck
- **Methods**: ok
### Class: TaskPlanner
- **Methods**: __init__, _tool_schema, _build_prompt, _parse_llm_plan, _fallback_plan, _clarification_plan, _enrich_plan, _normalize_steps
### Class: PluginCatalog
- **Methods**: __init__, summary
### Class: PluginManifest
- **Methods**: 
### Class: PluginManifestError
- **Methods**: 
### Class: BackgroundMonitor
- **Methods**: __init__
### Class: NotificationManager
- **Methods**: notify, schedule_reminder
### Class: RiskLevel
- **Methods**: 
### Class: Capability
- **Methods**: name, is_write_operation, risk_level, schema
### Class: FunctionCapability
- **Methods**: __init__
### Class: DesktopCapability
- **Methods**: __init__
### Class: CapabilityRegistry
- **Methods**: __init__, register, get, registered_tools, reset_call_count, get_observations, clear_observations, load_plugins
### Class: AuthUser
- **Methods**: 
### Class: AuthManager
- **Methods**: __init__, _connect, _init_db, bootstrap_admin_from_env, user_count, create_user, authenticate, create_api_token, verify_api_token, sign_session, verify_session, make_csrf_token, verify_csrf_token, hash_password, verify_password, _sign, _token_hash
### Class: PythonSearchEngine
- **Methods**: __init__, should_skip, is_binary, search_file_content, worker, run
### Class: ToolResult
- **Methods**: to_reflection_payload
### Class: VisionTool
- **Methods**: __init__, _get, analyze, _call_llava
### Class: SupportsQuickLLM
- **Methods**: 
### Class: SearchSettings
- **Methods**: from_sources
### Class: SearchResult
- **Methods**: 
### Class: WebToolService
- **Methods**: __init__, _provider_chain, _search_with_ddgs, _search_with_tavily, _scrape_page, _format_search_output
### Class: ToolResult
- **Methods**: to_llm_string, __repr__
### Class: IntegrationRiskLevel
- **Methods**: 
### Class: AudioInputSerde
- **Methods**: serialize, deserialize
### Class: AudioInputMixin
- **Methods**: audio_input, _audio_input, dg
### Class: AudioPlayer
- **Methods**: __init__, __enter__, __exit__, play, is_available
### Class: TranscriptResult
- **Methods**: 
### Class: STT
- **Methods**: __init__, _init, is_ready, _is_speech, capture_and_transcribe
### Class: SpeechToText
- **Methods**: __init__, _get, _choose_backend, _record_and_transcribe, _record_and_transcribe_google
### Class: TTS
- **Methods**: __init__, _init_backend, is_speaking, speak, stop, _speak_sentence
### Class: TextToSpeech
- **Methods**: __init__, _get, _engine_chain
### Class: _TextToSpeechStub
- **Methods**: __init__
### Class: VoiceLayer
- **Methods**: __init__
### Class: VoiceLoop
- **Methods**: __init__
### Class: WakeWordDetector
- **Methods**: __init__, _get, _fire_wake, _fire_cancel, _wait_blocking, stop

## External Dependencies
streamlit.proto.Common_pb2, base64, core.llm.client, core.tools.system_automation, core.runtime.import_validator, contextvars, pytesseract, core.runtime.container, core.tools.fast_search_tool, streamlit.errors, streamlit.runtime.scriptrunner, edge_tts, core.controller.web_search, streamlit.runtime.uploaded_file_manager, dataclasses, core.autonomy.autonomy_governor, core.controller.llm_dispatcher, psutil, core.llm.telemetry, pyttsx3, subprocess, httpx, core.desktop.observation, html, traceback, core.controller.intent_handlers, streamlit.elements.lib.layout_utils, core.memory.retriever, health, numpy, faster_whisper, core.automation.scan_rules, argparse, typing, core.planner.planner, core.llm.model_router, core.controller.intents, re, math, csv, core.controller.goal_runner, streamlit.elements.lib.file_uploader_utils, core.llm.model_spec, core.tools.builtin_tools, bs4, shlex, serial_controller, json, core.state_machine, abc, concurrent.futures, core.executor.engine, core.logging.logger, serial, ctypes, fnmatch, confidence, core.base_controller, core.voice.tts, asyncio, os, pathlib, datetime, core.voice.stt, core.proactive.notifier, plyer, sounddevice, core.desktop.contracts, core.tools.path_utils, pvrecorder, core.automation.scan_pipeline, core.agent.agent_loop, secrets, core.memory.code_indexer, core.controller.automation_manager, fpdf, copy, core.tools.web_tools, streamlit.delta_generator, core.proactive.background_monitor, core.capability.base, core.voice.voice_layer, torch.nn.functional, urllib.parse, core.llm.defaults, hashlib, parsedatetime, bcrypt, streamlit.proto.AudioInput_pb2, core.controller_v2, shutil, logging, streamlit.elements.widgets.file_uploader, contextlib, sqlite3, core.memory.embeddings, chromadb.config, configparser, cv2, aiohttp, PIL, textwrap, core.controller.llm_orchestrator, yaml, core.controller.complexity_scorer, core.context.context, sys, core.controller.intent_router, core.ops.production, pvporcupine, core.autonomy.risk_evaluator, __future__, core.memory.code_indexer_service, core.voice.wake_word, struct, queue, importlib.util, core.config, functools, core.memory.sqlite_storage, core.desktop.mission, core.runtime.event_bus, core.metrics.confidence, requests, chromadb, markdown, duckduckgo_search, streamlit.elements.lib.form_utils, streamlit.elements.lib.utils, device_registry, threading, core.tools.screen, uuid, core.controller.request_rules, urllib.request, core.desktop.shortcuts, time, core.llm.cloud_client, torch, streamlit.runtime.metrics_util, speech_recognition, core.tools.vision, core.memory.hybrid_memory, aiosqlite, ast, core.tools.gui_control, core.memory.context_compressor, hmac, core.memory.semantic_memory, core.autonomy.goal_manager, core.config.defaults, core.desktop.actions, core.controller.memory_subsystem, core.hardware.serial_controller, pandas, atexit, core.runtime.paths, pyperclip, core.profile, streamlit.runtime.state, core.registry.registry, core.executor.dag, enum, io, core.hardware.device_registry, core.tools.universal_converter, core.automation.rag_ingester, core.synthesis, core.types.common, sentence_transformers, pygetwindow, core.llm.ollama_client, logging.handlers, core.memory.sqlite_pool, core.voice.voice_loop, core.autonomy.scheduler, core.tools.hardware_tools, core.security.auth, platform, core.automation.payload_extractor, inspect, collections, pyautogui, tempfile, core.runtime.bootstrap, core.controller.services, pypdf, core.automation.live_automation, streamlit.elements.lib.policies