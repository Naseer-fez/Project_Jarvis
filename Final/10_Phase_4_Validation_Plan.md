# PHASE 4: MULTI-AGENT VALIDATION PLAN

## ACTIVE AGENTS
- Validator Agent
- Risk Assessment Agent
- Dependency Analysis Agent

## VALIDATION PROTOCOL
Before Phase 5 implementation begins, the recovery strategies outlined in Phase 3 must be challenged:

### Challenge 1: The ABC Controller Refactor
- **Risk Assessor**: Modifying the Controller instantiation to enforce `BaseController` might break legacy plugins that instantiate their own ad-hoc controllers.
- **Validation Requirement**: Grep/scan entire `plugins/` and `workflows/` directory for `_load_controller_class` or custom controller instantiations. Prove that no external logic bypasses the `main.py` entrypoint.

### Challenge 2: AsyncExitStack Implementation
- **Risk Assessor**: Implementing strict teardown might expose previously hidden errors in database drivers (e.g., ChromaDB failing to close gracefully).
- **Validation Requirement**: Write an isolated teardown test script for `chroma_db` and `dashboard` to verify they can handle explicit `.close()` or cancellation without throwing secondary exceptions.

### Challenge 3: Pydantic Schema Migration
- **Risk Assessor**: `jarvis.ini` might contain undocumented, loosely typed keys that are accessed dynamically via `config.get()`. A strict schema will cause immediate crashes.
- **Validation Requirement**: Extract all `config.get` calls from the codebase via `grep_search` and compile a comprehensive schema. No key can be left out.

**Status**: Validation criteria set. Requires automated scanning prior to Phase 5 execution.
