import configparser
import pytest
from core.runtime.container import ServiceContainer
from core.controller.services import build_controller_services


class MockMemory:

    def __init__(self, db_path, chroma_path=None, model_name=None):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.custom_flag = True

    def set_llm(self, llm, enable_context_titles=True):
        self.llm = llm


def test_service_container_basic_flow():
    container = ServiceContainer()

    # Test class registration
    container.register("memory", MockMemory)
    assert container.has("memory")

    # Resolve
    mem = container.resolve("memory", db_path="test.db")
    assert isinstance(mem, MockMemory)
    assert mem.db_path == "test.db"
    assert mem.custom_flag

    # Test instance registration
    class Dummy:
        pass

    dummy_inst = Dummy()
    container.register_instance("dummy", dummy_inst)
    assert container.resolve("dummy") is dummy_inst


def test_build_controller_services_respects_container_overrides():
    config = configparser.ConfigParser()
    config.add_section("memory")
    config.set("memory", "db_path", "test_memory.db")
    config.add_section("ollama")
    config.set("ollama", "base_url", "http://localhost:11434")

    container = ServiceContainer()
    # Override the "memory" service before building
    container.register("memory", MockMemory)

    # Build services using the pre-configured container
    settings, services = build_controller_services(config, container=container)

    # Assert container is exposed on services
    assert services.container is container

    # Assert that the custom memory mock was resolved instead of HybridMemory
    assert isinstance(services.memory, MockMemory)
    assert services.memory.custom_flag
    assert services.memory.db_path == settings.db_path


def test_dynamic_autonomy_and_risk_registration():
    from core.autonomy.autonomy_governor import AutonomyGovernor
    from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel
    from core.execution.dispatcher import Dispatcher
    from core.agentic.autonomy_policy import AutonomyPolicy

    # Test AutonomyGovernor dynamic registration
    gov = AutonomyGovernor(level=2)  # READ_ONLY
    # Default behavior: custom_tool is unknown and blocked
    allowed, reason = gov.can_execute("custom_read_tool")
    assert not allowed
    assert "blocked by default" in reason.lower()

    # Register as read-only
    gov.register_read_only_tool("custom_read_tool")
    allowed, reason = gov.can_execute("custom_read_tool")
    assert allowed
    assert "read-only tool" in reason.lower()

    # Register as write tool (blocked at level 2)
    gov.register_write_tool("custom_write_tool")
    allowed, reason = gov.can_execute("custom_write_tool")
    assert not allowed
    assert "blocked at level" in reason.lower()

    # Escalate to level 3 (WRITE_WITH_CONFIRM)
    gov.escalate(3)
    allowed, reason = gov.can_execute("custom_write_tool")
    assert allowed
    assert "approved at level_3" in reason.lower()

    # Test RiskEvaluator dynamic registration
    evaluator = RiskEvaluator()
    # Default behavior: unknown action treated as HIGH risk
    res = evaluator.evaluate(["custom_risky_action"])
    assert res.level == RiskLevel.HIGH

    # Register as CRITICAL (forbidden)
    evaluator.register_critical_action("custom_risky_action")
    res = evaluator.evaluate(["custom_risky_action"])
    assert res.level == RiskLevel.CRITICAL
    assert res.is_blocked

    # Register as LOW
    evaluator.register_low_action("custom_risky_action")
    res = evaluator.evaluate(["custom_risky_action"])
    assert res.level == RiskLevel.LOW

    # Test Dispatcher dynamic risk registration
    policy = AutonomyPolicy(None)
    dispatcher = Dispatcher(
        autonomy_policy=policy,
        reflection_engine=None,
    )
    # Default unknown core tool risk is 1.0 (highest score)
    from core.execution.dispatch_rules import core_or_desktop_risk_score
    score = core_or_desktop_risk_score("custom_tool", dispatcher.core_risk_registry)
    assert score == 1.0

    # Register customized risk score
    dispatcher.register_core_tool_risk("custom_tool", 0.25)
    score = core_or_desktop_risk_score("custom_tool", dispatcher.core_risk_registry)
    assert score == 0.25


def test_integration_auto_registration_safety_rules():
    from typing import Any
    from core.autonomy.autonomy_governor import AutonomyGovernor
    from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel
    from core.execution.dispatcher import Dispatcher
    from core.agentic.autonomy_policy import AutonomyPolicy
    from integrations.registry import IntegrationRegistry
    from integrations.base import BaseIntegration

    class MockIntegration(BaseIntegration):
        name = "mock_integration"
        description = "Mock integration for testing"

        def is_available(self) -> bool:
            return True

        def get_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "mock_read_action",
                    "risk": "low",
                },
                {
                    "name": "mock_write_action",
                    "risk": "confirm",
                },
                {
                    "name": "mock_med_action",
                    "risk": "medium",
                }
            ]

        async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"success": True}

    registry = IntegrationRegistry()
    registry.register(MockIntegration())

    gov = AutonomyGovernor(level=2)  # READ_ONLY
    evaluator = RiskEvaluator()
    dispatcher = Dispatcher(autonomy_policy=AutonomyPolicy(None), reflection_engine=None)

    # Dynamic registration
    registry.register_safety_rules(gov, evaluator, dispatcher)

    # 1. Verify AutonomyGovernor
    allowed, _ = gov.can_execute("mock_read_action")
    assert allowed  # Read-only is allowed at level 2
    allowed, _ = gov.can_execute("mock_write_action")
    assert not allowed  # Write requires level 3

    # 2. Verify RiskEvaluator
    res = evaluator.evaluate(["mock_read_action"])
    assert res.level == RiskLevel.LOW
    res = evaluator.evaluate(["mock_write_action"])
    assert res.level == RiskLevel.CONFIRM
    res = evaluator.evaluate(["mock_med_action"])
    assert res.level == RiskLevel.MEDIUM

    # 3. Verify Dispatcher
    from core.execution.dispatch_rules import integration_risk_score
    assert integration_risk_score("mock_read_action", dispatcher.integration_risk_registry) == 0.1
    assert integration_risk_score("mock_med_action", dispatcher.integration_risk_registry) == 0.6
    assert integration_risk_score("mock_write_action", dispatcher.integration_risk_registry) == 0.8


def test_service_container_edge_cases():
    import pytest
    container = ServiceContainer()

    # 1. Unregistered resolution raises ValueError
    with pytest.raises(ValueError, match="is not registered"):
        container.resolve("non_existent_service")

    # 2. Re-registration invalidates singleton cache
    class ServiceA:
        def __init__(self, val=1):
            self.val = val

    container.register("service_a", ServiceA, is_singleton=True)
    inst1 = container.resolve("service_a", val=10)
    assert inst1.val == 10

    # Resolve again (returns cached singleton)
    inst2 = container.resolve("service_a", val=20)
    assert inst2 is inst1
    assert inst2.val == 10

    # Re-register
    container.register("service_a", ServiceA, is_singleton=True)
    # Should resolve a new instance now since cache was invalidated
    inst3 = container.resolve("service_a", val=30)
    assert inst3 is not inst1
    assert inst3.val == 30

    # 3. Factory function registration
    def my_factory():
        return "factory_output"

    container.register("factory_service", my_factory, is_singleton=False)
    assert container.resolve("factory_service") == "factory_output"

    # 4. Check container reset clears registrations
    container.reset()
    assert not container.has("service_a")
    with pytest.raises(ValueError):
        container.resolve("service_a")


def test_autonomy_governor_edge_cases():
    from core.autonomy.autonomy_governor import AutonomyGovernor, AutonomyLevel

    # Initialise at SUGGEST_ONLY (level 1)
    gov = AutonomyGovernor(level=1)

    # Escalation bounds checks
    assert not gov.escalate(4)  # Cannot escalate above level 3
    assert gov.level == AutonomyLevel.SUGGEST_ONLY  # Level unchanged

    assert gov.escalate(3)
    assert gov.level == AutonomyLevel.WRITE_WITH_CONFIRM

    # Dynamic classification precedence check
    # Register as write tool first
    gov.register_write_tool("some_dual_tool")
    assert "some_dual_tool" in gov.write_tools
    assert "some_dual_tool" not in gov.read_only_tools

    # Overwrite as read-only tool
    gov.register_read_only_tool("some_dual_tool")
    assert "some_dual_tool" in gov.read_only_tools
    assert "some_dual_tool" not in gov.write_tools

    # Re-overwrite as write tool
    gov.register_write_tool("some_dual_tool")
    assert "some_dual_tool" in gov.write_tools
    assert "some_dual_tool" not in gov.read_only_tools

    # Unknown tools are blocked by default
    allowed, reason = gov.can_execute("unknown_arbitrary_tool")
    assert not allowed
    assert "blocked by default" in reason

    # requires_confirmation checks
    # Read-only tools do not require confirmation
    gov.register_read_only_tool("safe_read_tool")
    assert not gov.requires_confirmation("safe_read_tool")
    # Write tools require confirmation at Level 3
    assert gov.requires_confirmation("some_dual_tool")


def test_risk_evaluator_edge_cases():
    from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel

    evaluator = RiskEvaluator()

    # Precedence check: an action registered as critical, then low, should be low
    evaluator.register_critical_action("test_action")
    res = evaluator.evaluate(["test_action"])
    assert res.level == RiskLevel.CRITICAL
    assert res.is_blocked

    evaluator.register_low_action("test_action")
    res = evaluator.evaluate(["test_action"])
    assert res.level == RiskLevel.LOW
    assert not res.is_blocked
    assert not res.requires_confirmation

    # Re-escalate to confirm
    evaluator.register_confirm_action("test_action")
    res = evaluator.evaluate(["test_action"])
    assert res.level == RiskLevel.CONFIRM
    assert res.requires_confirmation

    # Test evaluation of multiple actions (should return max risk)
    evaluator.register_low_action("action_low")
    evaluator.register_confirm_action("action_confirm")
    evaluator.register_critical_action("action_critical")

    res = evaluator.evaluate(["action_low", "action_confirm"])
    assert res.level == RiskLevel.CONFIRM  # max(LOW, CONFIRM) = CONFIRM

    res = evaluator.evaluate(["action_low", "action_confirm", "action_critical"])
    assert res.level == RiskLevel.CRITICAL  # max(LOW, CONFIRM, CRITICAL) = CRITICAL


def test_integration_registry_auto_registration_edge_cases():
    from typing import Any
    from core.autonomy.autonomy_governor import AutonomyGovernor
    from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel
    from core.execution.dispatcher import Dispatcher
    from core.agentic.autonomy_policy import AutonomyPolicy
    from integrations.registry import IntegrationRegistry
    from integrations.base import BaseIntegration

    class IncompleteIntegration(BaseIntegration):
        name = "incomplete"
        description = "Has tools with missing or invalid risk metadata"

        def is_available(self) -> bool:
            return True

        def get_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "missing_risk_tool",
                    # No risk key
                },
                {
                    "name": "invalid_risk_tool",
                    "risk": "extremely_high_risk_value",  # Unexpected value
                }
            ]

        async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"success": True}

    registry = IntegrationRegistry()
    registry.register(IncompleteIntegration())

    gov = AutonomyGovernor(level=2)  # READ_ONLY
    evaluator = RiskEvaluator()
    dispatcher = Dispatcher(autonomy_policy=AutonomyPolicy(None), reflection_engine=None)

    # Dynamic registration
    registry.register_safety_rules(gov, evaluator, dispatcher)

    # Tools missing risk or having invalid risk metadata must fall back to Write/Confirm safely
    # Autonomy check:
    allowed, _ = gov.can_execute("missing_risk_tool")
    assert not allowed  # fallback is WRITE, so blocked at level 2
    allowed, _ = gov.can_execute("invalid_risk_tool")
    assert not allowed  # fallback is WRITE, so blocked at level 2

    # Risk check:
    res = evaluator.evaluate(["missing_risk_tool"])
    assert res.level == RiskLevel.CONFIRM  # fallback risk is CONFIRM
    res = evaluator.evaluate(["invalid_risk_tool"])
    assert res.level == RiskLevel.CONFIRM  # fallback risk is CONFIRM

    # Dispatcher check:
    from core.execution.dispatch_rules import integration_risk_score
    assert integration_risk_score("missing_risk_tool", dispatcher.integration_risk_registry) == 0.8
    assert integration_risk_score("invalid_risk_tool", dispatcher.integration_risk_registry) == 0.8


def test_event_bus_basic_pub_sub():
    from core.runtime.event_bus import EventBus

    bus = EventBus()
    received = []

    def sync_cb(data):
        received.append(("sync", data))

    async def async_cb(data):
        received.append(("async", data))

    # Subscribe
    bus.subscribe("test_event", sync_cb)
    bus.subscribe("test_event", async_cb)

    # Publish sync
    bus.publish("test_event", "hello")
    # Wait slightly if any async loop task was scheduled
    assert ("sync", "hello") in received

    # Unsubscribe sync
    bus.unsubscribe("test_event", sync_cb)
    received.clear()

    # Publish again
    bus.publish("test_event", "world")
    assert ("sync", "world") not in received


@pytest.mark.asyncio
async def test_state_machine_publishes_to_event_bus():
    from core.runtime.event_bus import EventBus
    from core.state_machine import StateMachine, State

    bus = EventBus()
    sm = StateMachine(event_bus=bus)

    events_received = []
    def track_transition(payload):
        events_received.append(payload)

    bus.subscribe("state_transition", track_transition)

    # Trigger transition
    sm.transition(State.PLANNING)
    assert len(events_received) == 1
    assert events_received[0]["old_state"] == "IDLE"
    assert events_received[0]["new_state"] == "PLANNING"


def test_di_resolved_agent_loop_and_desktop_bridge():
    from core.runtime.container import ServiceContainer
    from core.agent.agent_loop import AgentLoopEngine

    container = ServiceContainer()

    # Register mocks
    class DummySM:
        pass
    class DummyPlanner:
        pass
    class DummyRouter:
        pass
    class DummyRisk:
        pass
    class DummyGov:
        pass
    class DummyLLM:
        pass
    class DummyBridge:
        pass

    mock_sm = DummySM()
    mock_planner = DummyPlanner()
    mock_router = DummyRouter()
    mock_risk = DummyRisk()
    mock_gov = DummyGov()
    mock_llm = DummyLLM()
    mock_bridge = DummyBridge()

    container.register_instance("state_machine", mock_sm)
    container.register_instance("task_planner", mock_planner)
    container.register_instance("tool_router", mock_router)
    container.register_instance("risk_evaluator", mock_risk)
    container.register_instance("autonomy_governor", mock_gov)
    container.register_instance("llm", mock_llm)
    container.register_instance("desktop_bridge", mock_bridge)

    # Resolve AgentLoopEngine using container
    loop_engine = AgentLoopEngine(container=container)
    assert loop_engine.sm is mock_sm
    assert loop_engine.planner is mock_planner
    assert loop_engine.router is mock_router
    assert loop_engine.risk is mock_risk
    assert loop_engine.gov is mock_gov
    assert loop_engine.llm is mock_llm
    assert loop_engine.desktop_bridge is mock_bridge




