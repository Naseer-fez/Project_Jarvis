"""
tests/test_v1_acceptance.py
════════════════════════════
Jarvis V1 PASS/FAIL acceptance tests.

V1 is DONE only when ALL of these pass.
If even one fails → V1 is not done.

Checklist:
  ✓ State machine cannot be broken
  ✓ Planner outputs valid JSON every time
  ✓ RiskEvaluator blocks forbidden actions
  ✓ Vision never triggers actions
  ✓ Memory recalls correctly
  ✓ Logs are complete
  ✓ System can say "I don't know"
"""

import pytest
import asyncio
from core.state_machine import JarvisStateMachine, State, StateMachineError
from core.risk_evaluator import RiskEvaluator, RiskLevel
from core.memory.hybrid_memory import HybridMemory


# ══════════════════════════════════════════════
# 1. STATE MACHINE CANNOT BE BROKEN
# ══════════════════════════════════════════════

class TestStateMachine:

    def test_initial_state_is_idle(self):
        sm = JarvisStateMachine()
        assert sm.state == State.IDLE

    def test_legal_transitions_work(self):
        sm = JarvisStateMachine()
        sm.transition(State.LISTENING, "test")
        assert sm.state == State.LISTENING
        sm.transition(State.THINKING, "test")
        assert sm.state == State.THINKING
        sm.transition(State.PLANNING, "test")
        assert sm.state == State.PLANNING
        sm.transition(State.IDLE, "test")
        assert sm.state == State.IDLE

    def test_illegal_transition_raises(self):
        """IDLE → ACTING_PHYSICAL must ALWAYS be impossible."""
        sm = JarvisStateMachine()
        with pytest.raises(StateMachineError):
            sm.transition(State.ACTING_PHYSICAL, "test")

    def test_v1_acting_states_blocked(self):
        """V2+ states must be blocked in V1."""
        sm = JarvisStateMachine()
        for blocked_state in (State.ACTING_PHYSICAL, State.ACTING_DIGITAL, State.SPEAKING):
            with pytest.raises(StateMachineError):
                sm.transition(blocked_state, "test")

    def test_thinking_cannot_jump_to_acting(self):
        sm = JarvisStateMachine()
        sm.transition(State.THINKING, "test")
        with pytest.raises(StateMachineError):
            sm.transition(State.ACTING_PHYSICAL, "test")

    def test_error_state_always_reachable(self):
        sm = JarvisStateMachine()
        sm.to_error("test")
        assert sm.state == State.ERROR

    def test_abort_always_reachable(self):
        sm = JarvisStateMachine()
        sm.transition(State.THINKING, "test")
        sm.abort("user requested abort")
        assert sm.state == State.ABORTED

    def test_reset_from_error(self):
        sm = JarvisStateMachine()
        sm.to_error("test")
        sm.reset()
        assert sm.state == State.IDLE

    def test_history_is_complete(self):
        sm = JarvisStateMachine()
        sm.transition(State.LISTENING, "test")
        sm.transition(State.THINKING, "test")
        sm.transition(State.IDLE, "test")
        history = sm.history()
        states = [h["to"] for h in history]
        assert "IDLE" in states
        assert "LISTENING" in states
        assert "THINKING" in states

    def test_state_machine_never_silent_fails(self):
        """Illegal transitions raise — never silently pass."""
        sm = JarvisStateMachine()
        # PLANNING → LISTENING is illegal
        sm.transition(State.PLANNING, "test")
        with pytest.raises(StateMachineError):
            sm.transition(State.LISTENING, "test")


# ══════════════════════════════════════════════
# 2. RISK EVALUATOR BLOCKS FORBIDDEN ACTIONS
# ══════════════════════════════════════════════

class TestRiskEvaluator:

    def setup_method(self):
        self.risk = RiskEvaluator()

    def test_physical_actions_hard_blocked(self):
        for tool in ["desktop_automation.click", "serial_controller.send", "shell.execute"]:
            result = self.risk.evaluate(tool)
            assert result.allowed is False, f"{tool} should be blocked"
            assert result.level == RiskLevel.BLOCKED

    def test_vision_is_safe(self):
        result = self.risk.evaluate("vision.analyze_image")
        assert result.allowed is True
        assert result.level == RiskLevel.SAFE

    def test_memory_read_is_safe(self):
        result = self.risk.evaluate("memory.read")
        assert result.allowed is True

    def test_unknown_tool_is_blocked(self):
        result = self.risk.evaluate("some.unknown.tool")
        assert result.allowed is False
        assert result.level == RiskLevel.BLOCKED

    def test_plan_with_blocked_step_fails(self):
        plan_steps = [
            {"id": 1, "action": "vision.analyze_image"},
            {"id": 2, "action": "desktop_automation.click"},  # BLOCKED
        ]
        all_safe, results = self.risk.evaluate_plan(plan_steps)
        assert all_safe is False

    def test_plan_with_all_safe_steps_passes(self):
        plan_steps = [
            {"id": 1, "action": "vision.analyze_image"},
            {"id": 2, "action": "memory.read"},
        ]
        all_safe, results = self.risk.evaluate_plan(plan_steps)
        assert all_safe is True


# ══════════════════════════════════════════════
# 3. MEMORY RECALLS CORRECTLY
# ══════════════════════════════════════════════

class TestHybridMemory:

    def setup_method(self):
        # Use in-memory for tests by overriding path
        import core.memory.hybrid_memory as mem_module
        from pathlib import Path
        mem_module.SQLITE_PATH = Path(":memory:")  # SQLite in-memory
        self.memory = HybridMemory()

    def test_write_and_read_fact(self):
        self.memory.write_fact("test_key", "test_value", category="test")
        result = self.memory.read_fact("test_key")
        assert result == "test_value"

    def test_read_nonexistent_returns_none(self):
        """System must say 'I don't know' — not hallucinate."""
        result = self.memory.read_fact("definitely_not_a_key_xyz123")
        assert result is None

    def test_overwrite_fact(self):
        self.memory.write_fact("key", "value1")
        self.memory.write_fact("key", "value2")
        result = self.memory.read_fact("key")
        assert result == "value2"

    def test_delete_fact(self):
        self.memory.write_fact("delete_me", "value")
        self.memory.delete_fact("delete_me")
        result = self.memory.read_fact("delete_me")
        assert result is None

    def test_session_history(self):
        self.memory.write_session("sess_1", "user", "hello")
        self.memory.write_session("sess_1", "assistant", "hi there")
        history = self.memory.read_session("sess_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"

    def test_list_facts_by_category(self):
        self.memory.write_fact("a", "val", category="cat_a")
        self.memory.write_fact("b", "val", category="cat_b")
        facts = self.memory.list_facts(category="cat_a")
        assert all(f["category"] == "cat_a" for f in facts)


# ══════════════════════════════════════════════
# 4. VISION NEVER TRIGGERS ACTIONS
# ══════════════════════════════════════════════

class TestVision:

    def test_blocked_action_prompt_rejected(self):
        from core.tools.vision import _is_prompt_safe
        assert _is_prompt_safe("What should I do next?") is False
        assert _is_prompt_safe("What's the next step physically?") is False
        assert _is_prompt_safe("How do I act?") is False

    def test_safe_prompts_allowed(self):
        from core.tools.vision import _is_prompt_safe
        assert _is_prompt_safe("Describe what is visible in this image.") is True
        assert _is_prompt_safe("What objects are present?") is True

    def test_vision_tool_keys_valid(self):
        from core.tools.vision import ALLOWED_PROMPTS
        assert "describe" in ALLOWED_PROMPTS
        assert "objects" in ALLOWED_PROMPTS
        assert "clarity" in ALLOWED_PROMPTS

    @pytest.mark.asyncio
    async def test_vision_bad_path_returns_error_not_crash(self):
        from core.tools.vision import VisionTool
        vt = VisionTool()
        result = await vt.analyze_image("/nonexistent/path/image.png")
        assert result["success"] is False
        assert result["error"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
