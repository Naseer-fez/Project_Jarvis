# Monolith Migration Report

## Duplicate Symbols Renamed
- `_TEXT_EXTENSIONS` in `core/automation/live_automation.py` renamed to `core_automation_live_automation__TEXT_EXTENSIONS`
- `_TEXT_EXTENSIONS` in `core/automation/payload_extractor.py` renamed to `core_automation_payload_extractor__TEXT_EXTENSIONS`
- `_IMAGE_EXTENSIONS` in `core/automation/live_automation.py` renamed to `core_automation_live_automation__IMAGE_EXTENSIONS`
- `_IMAGE_EXTENSIONS` in `core/automation/payload_extractor.py` renamed to `core_automation_payload_extractor__IMAGE_EXTENSIONS`
- `_VIDEO_EXTENSIONS` in `core/automation/live_automation.py` renamed to `core_automation_live_automation__VIDEO_EXTENSIONS`
- `_VIDEO_EXTENSIONS` in `core/automation/payload_extractor.py` renamed to `core_automation_payload_extractor__VIDEO_EXTENSIONS`
- `_normalize_text` in `core/automation/live_automation.py` renamed to `core_automation_live_automation__normalize_text`
- `_normalize_text` in `core/automation/payload_extractor.py` renamed to `core_automation_payload_extractor__normalize_text`
- `_truncate` in `core/automation/live_automation.py` renamed to `core_automation_live_automation__truncate`
- `_truncate` in `core/automation/payload_extractor.py` renamed to `core_automation_payload_extractor__truncate`
- `_utcnow` in `core/autonomy/goal_manager.py` renamed to `core_autonomy_goal_manager__utcnow`
- `_utcnow` in `core/autonomy/scheduler.py` renamed to `core_autonomy_scheduler__utcnow`
- `RiskLevel` in `core/autonomy/risk_evaluator.py` renamed to `core_autonomy_risk_evaluator_RiskLevel`
- `RiskLevel` in `core/registry/base.py` renamed to `core_registry_base_RiskLevel`
- `Capability` in `core/capability/base.py` renamed to `core_capability_base_Capability`
- `Capability` in `core/registry/base.py` renamed to `core_registry_base_Capability`
- `_normalize_tool_result` in `core/capability/base.py` renamed to `core_capability_base__normalize_tool_result`
- `_normalize_tool_result` in `core/desktop/actions.py` renamed to `core_desktop_actions__normalize_tool_result`
- `load_config` in `core/config/__init__.py` renamed to `core_config___init___load_config`
- `load_config` in `core/runtime/bootstrap.py` renamed to `core_runtime_bootstrap_load_config`
- `PROJECT_ROOT` in `core/desktop/shortcuts.py` renamed to `core_desktop_shortcuts_PROJECT_ROOT`
- `PROJECT_ROOT` in `core/runtime/paths.py` renamed to `core_runtime_paths_PROJECT_ROOT`
- `PROJECT_ROOT` in `core/tools/web_tools.py` renamed to `core_tools_web_tools_PROJECT_ROOT`
- `DEFAULT_MODEL` in `core/llm/defaults.py` renamed to `core_llm_defaults_DEFAULT_MODEL`
- `DEFAULT_MODEL` in `core/memory/embeddings.py` renamed to `core_memory_embeddings_DEFAULT_MODEL`
- `DEFAULT_MODEL` in `core/memory/semantic_memory.py` renamed to `core_memory_semantic_memory_DEFAULT_MODEL`
- `DEFAULT_THRESHOLD` in `core/memory/context_compressor.py` renamed to `core_memory_context_compressor_DEFAULT_THRESHOLD`
- `DEFAULT_THRESHOLD` in `core/memory/semantic_memory.py` renamed to `core_memory_semantic_memory_DEFAULT_THRESHOLD`
- `DEFAULT_CONFIG_PATH` in `core/runtime/bootstrap.py` renamed to `core_runtime_bootstrap_DEFAULT_CONFIG_PATH`
- `DEFAULT_CONFIG_PATH` in `core/tools/web_tools.py` renamed to `core_tools_web_tools_DEFAULT_CONFIG_PATH`
- `_require_pyautogui` in `core/tools/gui_control.py` renamed to `core_tools_gui_control__require_pyautogui`
- `_require_pyautogui` in `core/tools/screen.py` renamed to `core_tools_screen__require_pyautogui`
- `ToolResult` in `core/tools/system_automation.py` renamed to `core_tools_system_automation_ToolResult`
- `ToolResult` in `core/types/common.py` renamed to `core_types_common_ToolResult`
- `_TOKEN_URL` in `integrations/clients/gmail.py` renamed to `integrations_clients_gmail__TOKEN_URL`
- `_TOKEN_URL` in `integrations/clients/google_calendar.py` renamed to `integrations_clients_google_calendar__TOKEN_URL`
- `_TOKEN_URL` in `integrations/clients/spotify.py` renamed to `integrations_clients_spotify__TOKEN_URL`

## Metrics
- Files merged: 145
- Classes merged: 156
- Functions merged: 194
- Final line count: 596
