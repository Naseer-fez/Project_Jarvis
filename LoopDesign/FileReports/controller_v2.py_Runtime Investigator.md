# Runtime Investigator Report: controller_v2.py

## Role Relevancy
Orchestrates memory, LLM sub-components, and acts as the top-level main execution pipeline/router for CLI/Voice/Desktop.

## Assumptions
- Uses `configparser` reading from `config/jarvis.ini`.
- Inputs over 4000 chars are truncated.
- CLI loop is `while True` reading input, processing, and responding via `sys.stdout`.
- Desktop automation assumes `allow_gui_automation` and `allow_app_launch` config parameters.
- Uses `_state_lock` (an asyncio.Lock) to ensure exchange thread safety.

## Schema & API Contracts
- **jarvis.ini sections**: `[execution]`, `[memory]`, `[automation]`, `[voice]`.
- **classify_request()**: Expected to return `{"class": str, "complexity": float, "skip_planner": bool, "route": str}`.

## Dependencies
- `core.base_controller.BaseController`
- `core.controller.intent_router.IntentRouter`
- Facades/Subsystems: `LLMDispatcher`, `GoalRunner`, `LLMOrchestrator`, `MemorySubsystem`, `AutomationManager`.
- `core.voice.voice_layer.VoiceLayer`

## Configuration Variables
- `db_path = "memory/memory.db"`
- `chroma_path = "data/chroma"`
- `model_name = DEFAULT_MODEL`
- `embedding_model = "all-MiniLM-L6-v2"`

## Prompts
- No raw prompts found.
