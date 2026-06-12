# Folder Analysis: cross_folder

## Folder Purpose
Contains components related to cross_folder.

## Findings
- **1** (High): There is a circular dependency between the core subsystem and the dashboard. The `core` module explicitly imports `dashboard.server` to bootstrap the web UI, while `dashboard.server` heavily imports from multiple `core.*` namespaces (such as `core.security.auth`, `core.ai_os`, and `core.plugins`).
- **2** (Critical): Integrations (e.g., GitHub, Notion, Calendar) are successfully loaded into an isolated `IntegrationRegistry` at startup. However, the core execution engine and task planner exclusively query `CapabilityRegistry` (`tool_router`). These two registries are never bridged, rendering the integrations invisible to the LLM.
- **3** (High): The shutdown sequence blocks the main asyncio event loop synchronously for up to 5 seconds when stopping the dashboard server.
- **4** (Medium): During a graceful shutdown, the system cancels the background `_goal_check_task` but fails to explicitly flush and persist in-memory goal mutations to disk.
- **5** (Medium): The `requirements.lock` file only resolves dependencies for the base runtime, omitting optional capabilities. Conversely, the PyInstaller build script (`jarvis.spec`) hardcodes these optional modules into its `hiddenimports` list.
- **6** (Low): The `audit` directory is orphaned. It contains secret scrubbing logic (`audit_logger.py`) that is disconnected from the active runtime. The actual logging system implements its own redundant data redaction.

## Risks & Dependencies
See full project roadmap.
