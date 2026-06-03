from typing import Any
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel
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


def test_integration_auto_registration_safety_rules():
    registry = IntegrationRegistry()
    registry.register(MockIntegration())

    gov = AutonomyGovernor(level=2)  # READ_ONLY
    evaluator = RiskEvaluator()

    # Dynamic registration
    registry.register_safety_rules(gov, evaluator)

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


def test_integration_registry_auto_registration_edge_cases():
    registry = IntegrationRegistry()
    registry.register(IncompleteIntegration())

    gov = AutonomyGovernor(level=2)  # READ_ONLY
    evaluator = RiskEvaluator()

    # Dynamic registration
    registry.register_safety_rules(gov, evaluator)

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
