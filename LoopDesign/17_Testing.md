# 17. Testing Subsystem Architecture

## 1. Intent and Philosophy (WHY does this subsystem exist?)
The Testing Subsystem exists to mathematically and programmatically guarantee the operational stability, state machine integrity, and boundary resilience of the Jarvis architecture. Jarvis is a non-deterministic, LLM-driven operating system controller built on highly asynchronous event loops, integrating deeply with hardware peripherals, local persistence, and external APIs. 
Given the compounding complexity of autonomous task execution, asynchronous DAGs, and physical side effects, a robust testing framework is not just for code correctness—it is an architectural imperative to prevent deadlocks, runaway execution, secret leakage, and catastrophic memory corruption.

## 2. Core Responsibilities (WHAT responsibility does it own?)
The testing suite (`tests/unit/`, `tests/integration/`) strictly governs the following responsibilities:
- **State Machine Verification:** Ensuring deterministic transitions and recovery protocols across the `AgentLoop`, `StateMachine`, and `DAGExecutor`.
- **Concurrency & Event Loop Health:** Validating that asynchronous callbacks (e.g., user interrupts) are handled correctly without blocking the main event loop or exhausting thread executors.
- **Dependency Injection & Bootstrapping:** Validating the `ServiceContainer` resolution, runtime validations, and startup sequences (`test_startup.py`, `test_service_container.py`).
- **Resilience & Fallback Validation:** Verifying offline fallbacks (`test_offline_fallback.py`), model routing (`test_model_router.py`), retry mechanisms (`test_ollama_retry.py`), and error recovery (`test_failure_recovery.py`).
- **Security & Sandboxing Validation:** Ensuring local execution bounds, profile permissions, and path limits (`test_path_sandboxing.py`, `test_permission_matrix.py`, `test_dashboard_security.py`).
- **Resource Management:** Ensuring cleanup of transient resources, database connections, and background monitors (`test_resource_cleanup.py`, `test_background_monitor.py`).

## 3. System Interactions & Workflows (HOW does it interact?)
The testing subsystem wraps the core OS components using boundary mocks and fixtures. 
- **Fixture Injection:** Utilizing `conftest.py`, the system intercepts and mocks real-world side effects. It replaces actual hardware interfaces (mic, camera, speaker), disk storage, and external API networks with deterministic stubs.
- **Event Bus Hooking:** Tests attach to the `EventBus` to verify that events (e.g., state transitions, telemetry, permission faults) are emitted correctly without needing to assert internal hidden state.
- **Integration Orchestration:** Tests invoke `ControllerV2` orchestration, simulating full execution pathways—from prompt ingestion, through offline-fallback decision trees, into the DAG execution, and finally returning synthetic telemetry, validating memory rollbacks and offline transitions.

## 4. Architectural Fragility (WHAT would break if removed?)
Removing the testing subsystem strips Jarvis of its safety harness, guaranteeing critical system failures:
- **Asynchronous Deadlocks:** Uncaught synchronization errors (e.g., passing sync lambdas to async callbacks as observed in `JARVIS-TESTS-001`) would hang the agent loop, blocking CI pipelines and crashing production instances upon user interrupts.
- **Silent Fallback Failures:** Without `test_offline_fallback.py` and `test_ollama_retry.py`, network drops could hang the thread pool entirely rather than gracefully failing over to the local Ollama instance.
- **Execution Runaway:** The DAG executor and autonomy layers (`test_autonomy.py`, `test_dag_executor.py`) could fall into infinite loops or spawn unbounded subprocesses, resulting in OOM crashes or OS instability.
- **Security Exploitation:** The loss of `test_path_sandboxing.py` and `test_permission_matrix.py` would allow unverified LLM payloads to overwrite arbitrary file paths outside the designated workspace.

## 5. Clean-Room Reconstruction (HOW would it be rebuilt from scratch?)
To rebuild the testing subsystem without source code, an engineer must enforce a rigorous, multi-layered deterministic boundary around the non-deterministic LLM core:

1. **Framework & Fixture Foundation:** Initialize a test framework (`pytest`) and establish a global `conftest.py` that hooks the `ServiceContainer`. All hardware components (Audio, Vision, Serial) and external network calls (HTTP requests) must be unconditionally mocked.
2. **Unit Layer - Concurrency & Execution:** Build tests explicitly targeting the `asyncio` event loop. Verify that the `EventBus`, `StateMachine`, and `DAGExecutor` can handle arbitrary delays, timeouts, and cancellations. Ensure that callbacks (e.g., `confirm_callback`) correctly await responses.
3. **Unit Layer - Isolation & Permissions:** Implement path sandboxing tests to prove directory traversal attacks fail. Implement permission matrix tests simulating varying user privileges against tool invocations.
4. **Integration Layer - Fallback & Recovery:** Orchestrate end-to-end simulated scenarios:
   - Provide a simulated "Network Disconnect" midway through an LLM request and assert the `ModelRouter` switches to local models.
   - Inject SQL integrity errors to verify database rollback logic in the `ResourceCleanup`.
5. **Adversarial Resiliency (Red Team Integration):** Implement specific chaos tests that simulate context window poisoning, thread-pool exhaustion (simulating infinite API rate limits), and malformed hardware telemetry inputs. Ensure secrets are sanitized in the resulting error logs.

## 6. Adversarial Analysis & Current Gaps
As identified in adversarial audits (`RedTeam_Audit_3.md`), testing must evolve beyond "happy path" AST analysis:
- Tests must explicitly simulate API Rate Limiting (HTTP 429) and ensure the worker thread pool does not stall and exhaust the event loop.
- Network endpoints and scraping tools must be tested against OOM payloads (e.g., massive binary blobs masquerading as HTML).
- Migration schema mismatches and lack of parameterization must be evaluated dynamically by intentionally injecting schema collisions during test setups.

## 7. Literal Fixture Definitions and Testing Thresholds

To ensure accurate clean-room reconstruction and proper testing thresholds (including configuration limits and state initialization bounds), the following literal schemas and fixtures MUST be integrated into the test suite's `conftest.py` exactly as written. These bypass non-deterministic side-effects and lock the system to specific thresholds.

### Test Configuration and Thresholds

```python
@pytest.fixture()
def mock_config() -> ConfigParser:
    """Returns a realistic ConfigParser with common sections pre-populated."""
    cfg = ConfigParser()
    cfg["general"] = {"session_name": "test-session"}
    cfg["agent"] = {"max_iterations": "10"}
    cfg["models"] = {"chat_model": "mistral:7b", "fallback_model": "mistral:7b"}
    cfg["proactive"] = {"cpu_alert_threshold": "90"}
    cfg["logging"] = {"level": "DEBUG", "file": "/tmp/jarvis-test.log"}
    cfg["voice"] = {"enabled": "false"}
    return cfg

@pytest.fixture()
def minimal_config() -> ConfigParser:
    """A bare-minimum ConfigParser with no sections — safe default."""
    return ConfigParser()
```

### Deterministic Isolation Boundaries

```python
import os
# We set these environment variables BEFORE importing application code to ensure
# offline mode and mock embeddings are strictly enforced.
os.environ["JARVIS_MOCK_EMBEDDINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Wrap build_controller_services to isolate sqlite/chromadb files during tests
import core.controller.services
original_build = core.controller.services.build_controller_services

def wrapped_build(config, *args, **kwargs):
    tmp_path = getattr(_current_test_tmp_path, "value", None)
    if tmp_path is not None:
        if "db_path" not in kwargs or kwargs["db_path"] == "memory/memory.db":
            kwargs["db_path"] = str(tmp_path / "test_memory.db")
        if "chroma_path" not in kwargs or kwargs["chroma_path"] == "data/chroma":
            kwargs["chroma_path"] = str(tmp_path / "test_chroma")
            
        try:
            if not config.has_section("memory"):
                config.add_section("memory")
            if not config.has_option("memory", "db_path"):
                config.set("memory", "db_path", str(tmp_path / "test_memory.db"))
            if not config.has_option("memory", "chroma_path"):
                config.set("memory", "chroma_path", str(tmp_path / "test_chroma"))
        except (AttributeError, TypeError):
            if hasattr(config, "__setitem__"):
                if "memory" not in config:
                    config["memory"] = {}
                if isinstance(config["memory"], dict):
                    if "db_path" not in config["memory"]:
                        config["memory"]["db_path"] = str(tmp_path / "test_memory.db")
                    if "chroma_path" not in config["memory"]:
                        config["memory"]["chroma_path"] = str(tmp_path / "test_chroma")
    return original_build(config, *args, **kwargs)

core.controller.services.build_controller_services = wrapped_build

# Wrap AuthManager.__init__ to isolate the auth database file during tests
import core.security.auth
original_auth_manager_init = core.security.auth.AuthManager.__init__

def wrapped_auth_manager_init(self, db_path, *args, **kwargs):
    tmp_path = getattr(_current_test_tmp_path, "value", None)
    if tmp_path is not None:
        db_path = tmp_path / "test_auth.db"
    original_auth_manager_init(self, db_path, *args, **kwargs)

core.security.auth.AuthManager.__init__ = wrapped_auth_manager_init
```

### Mocking Schemas

```python
@pytest.fixture()
def mock_llm() -> MagicMock:
    """Returns a mock LLM that simulates common responses."""
    llm = MagicMock()
    llm.complete = MagicMock(
        return_value='{"communication_style": {"value": "casual", "confidence": 0.8}}'
    )
    llm.complete_json = MagicMock(return_value={})
    return llm

@pytest.fixture()
def mock_controller() -> MagicMock:
    """Returns a mock Jarvis controller."""
    ctrl = MagicMock()
    ctrl.process = MagicMock(return_value="test response")
    ctrl.session_id = "test-session"
    return ctrl
```
