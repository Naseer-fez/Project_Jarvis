import asyncio
import configparser
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.agent.agent_loop import AgentLoopEngine
from core.state_machine import StateMachine, State
from core.context.context import TaskExecutionContext
from unittest.mock import MagicMock

from core.capability.base import ToolObservation
from core.introspection.health import HealthCheck, HealthReport, HealthStatus
from core.runtime.bootstrap import ExitCode
from core.runtime.entrypoint import async_run
from core.runtime.container import ServiceContainer
from core.executor.engine import DAGExecutor

class MockToolRouter:
    def reset_call_count(self):
        pass
    async def execute(self, action, params):
        return ToolObservation(tool_name=action, arguments=params, execution_status="success", output_summary="ok", error_message="")

@pytest.fixture
def mock_dependencies():
    return {
        "memory": MagicMock(),
        "llm": MagicMock(),
        "risk_evaluator": MagicMock(),
        "autonomy_governor": MagicMock(),
    }

@pytest.mark.asyncio
async def test_runtime_agent_loop_execution(mock_dependencies):
    """Test the full agent loop running a simple simulated DAG plan."""
    
    # We mock the planner to return a simple plan
    mock_planner = MagicMock()
    
    container = ServiceContainer()
    container.register("dag_executor", DAGExecutor, is_singleton=False)
    container.register("state_machine", StateMachine, is_singleton=False)

    agent_loop = AgentLoopEngine(
        llm=mock_dependencies["llm"],
        task_planner=mock_planner,
        tool_router=MockToolRouter(),  # type: ignore[arg-type]
        risk_evaluator=mock_dependencies["risk_evaluator"],
        autonomy_governor=mock_dependencies["autonomy_governor"],
        container=container
    )
    
    # Simulate risk evaluator passing
    mock_dependencies["risk_evaluator"].evaluate_plan = MagicMock(return_value=type("RiskResult", (), {"level": type("L", (), {"label": lambda: "low"}), "blocking_actions": set(), "confirm_actions": set(), "high_risk_actions": set(), "is_blocked": False, "requires_confirmation": False})())
    mock_dependencies["autonomy_governor"].can_execute = MagicMock(return_value=(True, ""))
    
    sm = StateMachine()
    context = TaskExecutionContext(task_id="test_task", trace_id="trace_1", state_machine=sm)
    
    from unittest.mock import AsyncMock
    mock_planner.plan = AsyncMock(return_value={
        "tools_required": True,
        "steps": [
            {"id": 1, "tool": "dummy_tool", "params": {}}
        ]
    })
    
    mock_dependencies["llm"].complete = AsyncMock(return_value="Reflection done")
    
    trace = await agent_loop.run(goal="Do something", context=context)
    
    assert trace is not None
    assert trace.success is True
    assert sm.state == State.IDLE  # Transitions to IDLE at the end
    
    # Check that execution actually happened
    assert len(trace.observations) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strict_health", "expected_exit_code"),
    [
        (False, ExitCode.OK),
        (True, ExitCode.STARTUP_ERROR),
    ],
)
async def test_async_run_health_check_strict_mode_controls_exit_code(
    monkeypatch,
    strict_health,
    expected_exit_code,
):
    config = configparser.ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "development")

    log = MagicMock()
    logger_mod = SimpleNamespace(
        setup=MagicMock(),
        get=MagicMock(return_value=log),
        audit=MagicMock(),
    )

    monkeypatch.setattr("core.runtime.entrypoint.load_config", lambda _: config)
    monkeypatch.setattr("core.runtime.entrypoint.apply_cli_overrides", lambda *_: None)
    monkeypatch.setattr("core.runtime.entrypoint._prepare_runtime_environment", lambda *_: None)
    monkeypatch.setattr("core.runtime.entrypoint._prepare_runtime_paths", lambda *_: None)
    monkeypatch.setattr("core.runtime.entrypoint._load_logger_module", lambda: logger_mod)
    monkeypatch.setattr(
        "core.runtime.entrypoint.run_lightweight_health_check",
        lambda *_: HealthReport(
            checks=[
                HealthCheck(
                    name="memory_sqlite",
                    status=HealthStatus.FAIL,
                    message="sqlite unavailable",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "core.runtime.entrypoint._load_controller_class",
        lambda: pytest.fail("health-check mode should not build a controller"),
    )

    args = SimpleNamespace(
        config="config/jarvis.ini",
        print_config=False,
        list_models=False,
        verify=False,
        voice=False,
        gui=False,
        dashboard=False,
        headless=False,
        health_check=True,
        strict_health=strict_health,
        shutdown_timeout=5.0,
        dashboard_host=None,
        dashboard_port=None,
    )

    assert await async_run(args) == expected_exit_code


@pytest.mark.asyncio
async def test_async_run_cancellation_still_shuts_down_controller(monkeypatch):
    config = configparser.ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "development")

    log = MagicMock()
    logger_mod = SimpleNamespace(
        setup=MagicMock(),
        get=MagicMock(return_value=log),
        audit=MagicMock(),
    )

    created: dict[str, object] = {}
    shutdown_requests: list[str] = []

    class FakeController:
        def __init__(self, config, voice):
            self.config = config
            self.voice = voice
            self.session_id = "test-session"
            self.start = AsyncMock()
            self.shutdown = AsyncMock()
            self.session_summary = MagicMock(return_value={"exchanges": 0})
            created["controller"] = self

    class FakeShutdownCoordinator:
        def __init__(self, _loop):
            self._event = asyncio.Event()

        def install_signal_handlers(self) -> None:
            return None

        def request_shutdown(self, signame: str = "manual") -> None:
            shutdown_requests.append(signame)
            self._event.set()

        async def wait(self) -> None:
            await self._event.wait()

    async def cancel_runtime_loop(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr("core.runtime.entrypoint.load_config", lambda _: config)
    monkeypatch.setattr("core.runtime.entrypoint.apply_cli_overrides", lambda *_: None)
    monkeypatch.setattr("core.runtime.entrypoint._prepare_runtime_environment", lambda *_: None)
    monkeypatch.setattr("core.runtime.entrypoint._prepare_runtime_paths", lambda *_: None)
    monkeypatch.setattr("core.runtime.entrypoint._load_logger_module", lambda: logger_mod)
    monkeypatch.setattr(
        "core.runtime.entrypoint.run_lightweight_health_check",
        lambda *_: HealthReport(),
    )
    monkeypatch.setattr("core.runtime.entrypoint._load_controller_class", lambda: FakeController)
    monkeypatch.setattr("core.runtime.entrypoint._load_integrations", lambda *_: {})
    monkeypatch.setattr(
        "core.runtime.entrypoint._run_startup_health_check",
        lambda *_args, **_kwargs: HealthReport(),
    )
    monkeypatch.setattr("core.runtime.entrypoint._run_runtime_loop", cancel_runtime_loop)

    args = SimpleNamespace(
        config="config/jarvis.ini",
        print_config=False,
        list_models=False,
        verify=False,
        voice=False,
        gui=False,
        dashboard=False,
        headless=False,
        health_check=False,
        strict_health=False,
        shutdown_timeout=5.0,
        dashboard_host=None,
        dashboard_port=None,
    )

    assert await async_run(args, shutdown_cls=FakeShutdownCoordinator) == ExitCode.OK

    controller = created["controller"]
    controller.start.assert_awaited_once()  # type: ignore[attr-defined]
    controller.shutdown.assert_awaited_once()  # type: ignore[attr-defined]
    assert shutdown_requests == ["cancelled"]
