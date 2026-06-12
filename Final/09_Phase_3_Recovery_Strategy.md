# PHASE 3: RECOVERY STRATEGY GENERATION

## ACTIVE AGENTS
- Recovery Coordination Agent
- Architecture Analysis Agent
- Refactor Planning Agent

## 1. CRITICAL RECOVERY: Async Lifecycles
- **Strategy**: Implement `AsyncExitStack` and explicit context managers for all long-running tasks. 
- **Action**: Replace `contextlib.suppress(Exception)` with precise error handling. Ensure `Controller` exposes a guaranteed `async def cleanup()` method.
- **Priority**: 1 (Catastrophic Risk Mitigation)

## 2. STRUCTURAL RECOVERY: Interface Contracts
- **Strategy**: Define a `BaseController` Abstract Base Class (ABC) in `core.interfaces`. All dynamically loaded controllers MUST inherit from this ABC.
- **Action**: Enforce `isinstance(controller, BaseController)` immediately after dynamic loading. Update type hints in `entrypoint.py` from `Any` to `BaseController`.
- **Priority**: 2 (Architecture Repair)

## 3. PERFORMANCE RECOVERY: Event Loop Protection
- **Strategy**: Audit `core.state_machine` and all `plugins` for blocking synchronous calls (e.g., `requests.get`, heavy file I/O).
- **Action**: Wrap blocking calls in `asyncio.to_thread()` or migrate to `aiohttp`/`aiofiles`. Introduce health-check heartbeat monitors to detect event loop stalling.
- **Priority**: 3 (Runtime Stability)

## 4. RELIABILITY RECOVERY: Configuration Schema Validation
- **Strategy**: Replace raw `configparser` reading with Pydantic configuration schemas.
- **Action**: Introduce `core.config.schema.py`. Validate `jarvis.ini` and environment variables on boot, failing explicitly with detailed validation errors before instantiating any subsystems.
- **Priority**: 4 (Maintainability & Resilience)
