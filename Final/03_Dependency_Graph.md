# Dependency Graph

## External Dependencies
- Python 3.x
- `asyncio` for main asynchronous orchestration
- Vector Databases: `chroma_db` (implied)
- Local Dependencies: `requirements.txt`, `requirements.lock`
- Integrations: Presumed 3rd party APIs, UI frameworks (Dash/Streamlit/Custom).

## Internal Coupling
- `main.py` -> `core.runtime.entrypoint` -> `core.runtime.bootstrap`
- `core.runtime` <- dependent on -> `core.introspection` (Health checks)
- `core.runtime` -> `core.runtime.dashboard_runtime`
- `core.controller_v2` is the central hub, acting as an orchestrator that pulls in `core.state_machine`, `core.plugins`, `core.integrations`.

## Circular Dependency Risks
- `core.runtime.bootstrap` loads classes dynamically to avoid Python module-level cyclic dependencies. While circumventing import errors, it indicates high coupling between `runtime`, `controller`, and `logging` modules.
