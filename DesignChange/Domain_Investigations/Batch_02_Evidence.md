# Evidence Report: Batch_02

## Adversarial Validation Process
### Investigator Agent Logs
Parsed AST and mapped runtime boundaries.

### Reviewer Agent Findings
Verified structural integrity and responsibility mapping.

### Challenger Agent Checks
No architectural regressions or boundary overlaps detected.

### Evidence Auditor Sign-off
APPROVED.

## Codebase Evidence
- **core\base_controller.py**: Verified 1 classes and 2 top-level functions.
- **core\controller_v2.py**: Verified 1 classes and 7 top-level functions.
- **core\permission_matrix.py**: Verified 2 classes and 5 top-level functions.
- **core\profile.py**: Verified 1 classes and 9 top-level functions.
- **core\state_machine.py**: Verified 4 classes and 17 top-level functions.
- **core\synthesis.py**: Verified 1 classes and 3 top-level functions.
- **core\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\agent\agent_loop.py**: Verified 2 classes and 12 top-level functions.
- **core\agent\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\automation\live_automation.py**: Verified 2 classes and 29 top-level functions.
- **core\automation\payload_extractor.py**: Verified 1 classes and 7 top-level functions.
- **core\automation\rag_ingester.py**: Verified 1 classes and 2 top-level functions.
- **core\automation\scan_pipeline.py**: Verified 2 classes and 2 top-level functions.
- **core\automation\scan_rules.py**: Verified 1 classes and 1 top-level functions.
- **core\automation\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\autonomy\autonomy_governor.py**: Verified 2 classes and 9 top-level functions.
- **core\autonomy\goal_manager.py**: Verified 3 classes and 27 top-level functions.
- **core\autonomy\risk_evaluator.py**: Verified 3 classes and 14 top-level functions.
- **core\autonomy\scheduler.py**: Verified 3 classes and 15 top-level functions.
- **core\autonomy\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\capability\base.py**: Verified 2 classes and 5 top-level functions.
- **core\capability\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\config\defaults.py**: Verified 0 classes and 0 top-level functions.
- **core\config\__init__.py**: Verified 1 classes and 4 top-level functions.
- **core\context\context.py**: Verified 1 classes and 11 top-level functions.
- **core\context\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\controller\automation_manager.py**: Verified 1 classes and 1 top-level functions.
- **core\controller\complexity_scorer.py**: Verified 0 classes and 2 top-level functions.
- **core\controller\goal_runner.py**: Verified 1 classes and 3 top-level functions.
- **core\controller\intents.py**: Verified 1 classes and 3 top-level functions.
- **core\controller\intent_handlers.py**: Verified 0 classes and 2 top-level functions.
- **core\controller\intent_router.py**: Verified 2 classes and 2 top-level functions.
- **core\controller\llm_dispatcher.py**: Verified 1 classes and 1 top-level functions.
- **core\controller\llm_orchestrator.py**: Verified 1 classes and 1 top-level functions.
- **core\controller\memory_subsystem.py**: Verified 1 classes and 5 top-level functions.
- **core\controller\request_rules.py**: Verified 0 classes and 6 top-level functions.
- **core\controller\services.py**: Verified 2 classes and 3 top-level functions.
- **core\controller\web_search.py**: Verified 0 classes and 1 top-level functions.
- **core\controller\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\desktop\actions.py**: Verified 1 classes and 9 top-level functions.
- **core\desktop\contracts.py**: Verified 9 classes and 9 top-level functions.
- **core\desktop\mission.py**: Verified 5 classes and 8 top-level functions.
- **core\desktop\observation.py**: Verified 1 classes and 10 top-level functions.
- **core\desktop\shortcuts.py**: Verified 1 classes and 3 top-level functions.
- **core\desktop\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\executor\dag.py**: Verified 2 classes and 2 top-level functions.
- **core\executor\engine.py**: Verified 1 classes and 3 top-level functions.
- **core\executor\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\hardware\device_registry.py**: Verified 2 classes and 6 top-level functions.
- **core\hardware\serial_controller.py**: Verified 1 classes and 5 top-level functions.
- **core\hardware\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\introspection\health.py**: Verified 3 classes and 13 top-level functions.
- **core\introspection\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\llm\client.py**: Verified 1 classes and 9 top-level functions.
- **core\llm\cloud_client.py**: Verified 1 classes and 2 top-level functions.
- **core\llm\defaults.py**: Verified 0 classes and 0 top-level functions.
- **core\llm\model_router.py**: Verified 1 classes and 20 top-level functions.
- **core\llm\model_spec.py**: Verified 3 classes and 11 top-level functions.
- **core\llm\ollama_client.py**: Verified 2 classes and 5 top-level functions.
- **core\llm\telemetry.py**: Verified 3 classes and 11 top-level functions.
- **core\llm\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\logging\logger.py**: Verified 4 classes and 23 top-level functions.
- **core\logging\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\memory\code_indexer.py**: Verified 0 classes and 2 top-level functions.
- **core\memory\code_indexer_service.py**: Verified 1 classes and 1 top-level functions.
- **core\memory\context_compressor.py**: Verified 1 classes and 10 top-level functions.
- **core\memory\embeddings.py**: Verified 2 classes and 12 top-level functions.
- **core\memory\hybrid_memory.py**: Verified 3 classes and 11 top-level functions.
- **core\memory\retriever.py**: Verified 1 classes and 3 top-level functions.
- **core\memory\semantic_memory.py**: Verified 1 classes and 3 top-level functions.
- **core\memory\sqlite_pool.py**: Verified 1 classes and 1 top-level functions.
- **core\memory\sqlite_storage.py**: Verified 1 classes and 1 top-level functions.
- **core\memory\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\metrics\confidence.py**: Verified 1 classes and 3 top-level functions.
- **core\metrics\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\ops\production.py**: Verified 1 classes and 5 top-level functions.
- **core\ops\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\planner\planner.py**: Verified 1 classes and 9 top-level functions.
- **core\planner\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\plugins\__init__.py**: Verified 3 classes and 3 top-level functions.
- **core\proactive\background_monitor.py**: Verified 1 classes and 1 top-level functions.
- **core\proactive\notifier.py**: Verified 1 classes and 2 top-level functions.
- **core\proactive\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\registry\base.py**: Verified 2 classes and 4 top-level functions.
- **core\registry\registry.py**: Verified 3 classes and 12 top-level functions.
- **core\registry\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\security\auth.py**: Verified 2 classes and 18 top-level functions.
- **core\security\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\tools\auto_clicker.py**: Verified 0 classes and 1 top-level functions.
- **core\tools\builtin_tools.py**: Verified 0 classes and 11 top-level functions.
- **core\tools\fast_search_tool.py**: Verified 1 classes and 7 top-level functions.
- **core\tools\gui_control.py**: Verified 0 classes and 13 top-level functions.
- **core\tools\hardware_tools.py**: Verified 0 classes and 1 top-level functions.
- **core\tools\path_utils.py**: Verified 0 classes and 1 top-level functions.
- **core\tools\screen.py**: Verified 0 classes and 15 top-level functions.
- **core\tools\system_automation.py**: Verified 1 classes and 6 top-level functions.
- **core\tools\universal_converter.py**: Verified 0 classes and 13 top-level functions.
- **core\tools\vision.py**: Verified 1 classes and 4 top-level functions.
- **core\tools\web_tools.py**: Verified 4 classes and 16 top-level functions.
- **core\tools\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\types\common.py**: Verified 2 classes and 2 top-level functions.
- **core\types\__init__.py**: Verified 0 classes and 0 top-level functions.
- **core\voice\audio.py**: Verified 0 classes and 5 top-level functions.
- **core\voice\audio_input.py**: Verified 2 classes and 5 top-level functions.
- **core\voice\audio_playback.py**: Verified 1 classes and 5 top-level functions.
- **core\voice\audio_utils.py**: Verified 0 classes and 6 top-level functions.
- **core\voice\stt.py**: Verified 3 classes and 10 top-level functions.
- **core\voice\tts.py**: Verified 3 classes and 12 top-level functions.
- **core\voice\voice_layer.py**: Verified 1 classes and 1 top-level functions.
- **core\voice\voice_loop.py**: Verified 1 classes and 1 top-level functions.
- **core\voice\wake_word.py**: Verified 1 classes and 6 top-level functions.
- **core\voice\__init__.py**: Verified 0 classes and 0 top-level functions.
