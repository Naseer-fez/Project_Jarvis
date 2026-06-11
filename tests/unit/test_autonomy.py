from core.autonomy.autonomy_governor import AutonomyGovernor, AutonomyLevel
from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel


def test_dynamic_autonomy_and_risk_registration():
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


def test_autonomy_governor_edge_cases():
    # Initialise at SUGGEST_ONLY (level 1)
    gov = AutonomyGovernor(level=1)

    # Escalation bounds checks
    assert not gov.escalate(5)  # Cannot escalate above level 4
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


def test_autonomy_governor_normalization_and_thread_safety():
    import threading

    # 1. Normalization check: register mixed-case/whitespace name
    gov = AutonomyGovernor(level=2)  # READ_ONLY
    gov.register_read_only_tool("  My_Custom_Read_Tool   ")
    
    # lookup should work regardless of spaces/case
    allowed, reason = gov.can_execute("my_custom_read_tool")
    assert allowed
    allowed, reason = gov.can_execute("MY_CUSTOM_READ_TOOL")
    assert allowed
    allowed, reason = gov.can_execute("  My_Custom_Read_Tool  ")
    assert allowed

    # 2. Thread safety check: run parallel lookups/writes
    errors = []
    
    def worker():
        try:
            for i in range(100):
                gov.register_read_only_tool(f"tool_{i}")
                gov.register_write_tool(f"write_tool_{i}")
                # Concurrent lookups
                gov.can_execute(f"tool_{i}")
                gov.can_execute("my_custom_read_tool")
                gov.requires_confirmation(f"write_tool_{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threading errors occurred: {errors}"


def test_risk_evaluator_registry_and_thread_safety():
    import threading
    from core.registry.registry import CapabilityRegistry
    
    # 1. Check capability registry integration
    registry = CapabilityRegistry()
    evaluator = RiskEvaluator(registry=registry)
    
    # Register a capability directly
    from core.autonomy.risk_evaluator import RiskLevel as RiskLvl
    from core.registry.registry import FunctionCapability
    
    def dummy_func():
        pass
        
    cap = FunctionCapability("special_action", dummy_func, risk_level=RiskLvl.CRITICAL, is_write=True)
    registry.register(cap)
    
    # RiskEvaluator should resolve this dynamically from registry
    res = evaluator.evaluate(["special_action"])
    assert res.level == RiskLvl.CRITICAL
    
    # 2. Thread safety check
    errors = []
    
    def worker():
        try:
            for i in range(100):
                evaluator.register_low_action(f"action_{i}")
                evaluator.register_critical_action(f"critical_{i}")
                res2 = evaluator.evaluate([f"action_{i}", f"critical_{i}", "special_action"])
                assert res2.level == RiskLvl.CRITICAL
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threading errors occurred: {errors}"
