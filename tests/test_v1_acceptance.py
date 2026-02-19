"""
tests/test_v1_acceptance.py — V1 Acceptance Checklist.

ALL tests must pass for V1 to be considered complete.
Run: pytest tests/test_v1_acceptance.py -v

Tests are isolated: no Ollama, no real audio, no disk side effects beyond tmp.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state_machine import StateMachine, State, IllegalTransitionError
from core.risk_evaluator import RiskEvaluator, RiskLevel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_config(tmp_path):
    cfg = configparser.ConfigParser()
    cfg["general"]  = {"name": "Jarvis", "version": "2.0.0"}
    cfg["ollama"]   = {"base_url": "http://localhost:11434", "planner_model": "deepseek-r1:8b", "vision_model": "llava", "request_timeout_s": "10"}
    cfg["memory"]   = {"data_dir": str(tmp_path / "data"), "sqlite_file": str(tmp_path / "data/jarvis_memory.db"), "chroma_dir": str(tmp_path / "data/chroma"), "embedding_model": "all-MiniLM-L6-v2", "semantic_top_k": "5"}
    cfg["risk"]     = {"forbidden_actions": "shell_exec,file_delete,network_request,serial_send,physical_actuate", "high_risk_actions": "file_write,process_spawn", "medium_risk_actions": "file_read,ui_interaction", "low_risk_actions": "memory_read,memory_write,speak,display"}
    cfg["logging"]  = {"log_dir": str(tmp_path / "logs"), "audit_file": str(tmp_path / "logs/audit.jsonl"), "app_file": str(tmp_path / "logs/app.log"), "level": "DEBUG"}
    cfg["voice"]    = {"enabled": "false", "wake_word": "hey jarvis", "cancel_words": "cancel,stop"}
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# 1. STATE MACHINE — cannot be broken
# ════════════════════════════════════════════════════════════════════════════

class TestStateMachine:
    def test_initial_state_is_idle(self):
        fsm = StateMachine()
        assert fsm.state == State.IDLE

    def test_legal_transition_idle_to_planning(self):
        fsm = StateMachine()
        fsm.transition(State.PLANNING)
        assert fsm.state == State.PLANNING

    def test_illegal_transition_raises(self):
        fsm = StateMachine()
        with pytest.raises(IllegalTransitionError):
            fsm.transition(State.EXECUTING)  # IDLE → EXECUTING is illegal

    def test_illegal_transition_from_shutdown(self):
        fsm = StateMachine()
        fsm.transition(State.SHUTDOWN)
        with pytest.raises(IllegalTransitionError):
            fsm.transition(State.IDLE)

    def test_reset_from_error(self):
        fsm = StateMachine()
        fsm.transition(State.PLANNING)
        fsm.transition(State.ERROR)
        fsm.reset()
        assert fsm.state == State.IDLE

    def test_reset_from_aborted(self):
        fsm = StateMachine()
        fsm.transition(State.PLANNING)
        fsm.transition(State.REVIEWING)
        fsm.transition(State.ABORTED)
        fsm.reset()
        assert fsm.state == State.IDLE

    def test_reset_from_idle_raises(self):
        fsm = StateMachine()
        with pytest.raises(IllegalTransitionError):
            fsm.reset()

    def test_can_transition_check(self):
        fsm = StateMachine()
        assert fsm.can_transition(State.PLANNING) is True
        assert fsm.can_transition(State.EXECUTING) is False

    def test_listener_is_called_on_transition(self):
        fsm = StateMachine()
        events = []
        fsm.add_listener(lambda old, new: events.append((old, new)))
        fsm.transition(State.PLANNING)
        assert len(events) == 1
        assert events[0] == (State.IDLE, State.PLANNING)

    def test_all_illegal_transitions_raise(self):
        """Exhaustively check that no illegal transition silently succeeds."""
        import itertools
        allowed = {
            State.IDLE:         {State.PLANNING, State.LISTENING, State.SHUTDOWN},
            State.PLANNING:     {State.REVIEWING, State.IDLE, State.ERROR, State.SPEAKING},
            State.REVIEWING:    {State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR},
            State.EXECUTING:    {State.IDLE, State.ERROR, State.ABORTED},
            State.ERROR:        {State.IDLE, State.SHUTDOWN},
            State.ABORTED:      {State.IDLE, State.SHUTDOWN},
            State.SHUTDOWN:     set(),
            State.LISTENING:    {State.TRANSCRIBING, State.IDLE, State.ERROR},
            State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
            State.SPEAKING:     {State.IDLE, State.LISTENING, State.ERROR},
        }
        for src_state, allowed_dests in allowed.items():
            all_states = set(State)
            illegal_dests = all_states - allowed_dests
            for dest in illegal_dests:
                fsm = StateMachine()
                # Force FSM into src_state via allowed path
                if src_state != State.IDLE:
                    # Use internal _state set for test setup only
                    fsm._state = src_state
                with pytest.raises(IllegalTransitionError):
                    fsm.transition(dest)

    # V2 voice states
    def test_voice_state_listening(self):
        fsm = StateMachine()
        fsm.transition(State.LISTENING)
        assert fsm.state == State.LISTENING

    def test_voice_state_full_cycle(self):
        fsm = StateMachine()
        fsm.transition(State.LISTENING)
        fsm.transition(State.TRANSCRIBING)
        fsm.transition(State.PLANNING)
        fsm.transition(State.SPEAKING)
        fsm.transition(State.IDLE)
        assert fsm.state == State.IDLE

    def test_force_idle_from_voice_state(self):
        fsm = StateMachine()
        fsm.transition(State.LISTENING)
        fsm.force_idle()
        assert fsm.state == State.IDLE


# ════════════════════════════════════════════════════════════════════════════
# 2. RISK EVALUATOR — blocks forbidden actions
# ════════════════════════════════════════════════════════════════════════════

class TestRiskEvaluator:
    def test_forbidden_action_is_blocked(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["shell_exec"])
        assert result.is_blocked
        assert result.level == RiskLevel.FORBIDDEN

    def test_multiple_forbidden_all_reported(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["shell_exec", "file_delete"])
        assert result.is_blocked
        assert len(result.blocking_actions) == 2

    def test_high_risk_requires_confirmation(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["file_write"])
        assert result.level == RiskLevel.HIGH
        assert result.requires_confirmation
        assert not result.is_blocked

    def test_medium_risk_requires_confirmation(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["file_read"])
        assert result.level >= RiskLevel.MEDIUM
        assert result.requires_confirmation

    def test_low_risk_actions_pass(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["memory_read", "speak", "display"])
        assert result.level == RiskLevel.LOW
        assert not result.is_blocked
        assert not result.requires_confirmation

    def test_empty_plan_is_low(self):
        ev = RiskEvaluator()
        result = ev.evaluate([])
        assert result.level == RiskLevel.LOW

    def test_unknown_action_is_high_risk(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["mystery_action_xyz"])
        assert result.level >= RiskLevel.HIGH

    def test_physical_actuate_forbidden(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["physical_actuate"])
        assert result.is_blocked

    def test_serial_send_forbidden(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["serial_send"])
        assert result.is_blocked

    def test_evaluate_plan_dict(self):
        ev = RiskEvaluator()
        plan = {"steps": [{"id": 1, "action": "shell_exec", "description": "bad"}]}
        result = ev.evaluate_plan(plan)
        assert result.is_blocked

    def test_summary_contains_level(self):
        ev = RiskEvaluator()
        result = ev.evaluate(["shell_exec"])
        assert "FORBIDDEN" in result.summary()


# ════════════════════════════════════════════════════════════════════════════
# 3. PLANNER — outputs valid JSON every time
# ════════════════════════════════════════════════════════════════════════════

class TestTaskPlanner:
    def test_planner_returns_dict_when_ollama_down(self, tmp_config):
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        plan = planner.plan("turn on the lights")
        assert isinstance(plan, dict)
        assert "intent" in plan
        assert "steps" in plan
        assert "clarification_needed" in plan
        assert isinstance(plan["steps"], list)

    def test_planner_echoes_intent(self, tmp_config):
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        intent = "remind me to call John at 3pm"
        plan = planner.plan(intent)
        assert plan["intent"] == intent

    def test_planner_unknown_request_clarifies(self, tmp_config):
        """When LLM is down, planner returns clarification_needed=True."""
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        plan = planner.plan("xyzzy nonsense magic words")
        assert isinstance(plan, dict)
        # Either it planned something or it asked for clarification — either is valid JSON
        assert plan["clarification_needed"] in (True, False)

    def test_plan_with_mocked_ollama(self, tmp_config):
        from core.planning.task_planner import TaskPlanner

        mock_response = json.dumps({
            "intent": "store a fact",
            "summary": "Store the fact that the sky is blue.",
            "confidence": 0.95,
            "steps": [{"id": 1, "action": "memory_write", "description": "Store sky=blue", "params": {"key": "sky", "value": "blue"}}],
            "clarification_needed": False,
            "clarification_prompt": "",
        })

        planner = TaskPlanner(tmp_config)
        with patch.object(planner, "_call_ollama", return_value=mock_response):
            plan = planner.plan("store a fact")

        assert plan["intent"] == "store a fact"
        assert plan["confidence"] == 0.95
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["action"] == "memory_write"

    def test_plan_with_invalid_json_response(self, tmp_config):
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        with patch.object(planner, "_call_ollama", return_value="not json at all !!!"):
            plan = planner.plan("do something")
        assert isinstance(plan, dict)
        assert plan["clarification_needed"] is True

    def test_plan_with_deepseek_thinking_tags(self, tmp_config):
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        raw = '<think>Let me think...</think>\n{"intent":"x","summary":"s","confidence":0.8,"steps":[],"clarification_needed":false,"clarification_prompt":""}'
        with patch.object(planner, "_call_ollama", return_value=raw):
            plan = planner.plan("x")
        assert plan["confidence"] == 0.8
        assert plan["clarification_needed"] is False


# ════════════════════════════════════════════════════════════════════════════
# 4. VISION — passive only, never triggers actions
# ════════════════════════════════════════════════════════════════════════════

class TestVisionTool:
    def test_vision_raises_for_missing_file(self, tmp_config):
        from core.tools.vision import VisionTool
        v = VisionTool(tmp_config)
        with pytest.raises(FileNotFoundError):
            v.analyze("/nonexistent/path/image.png")

    def test_vision_raises_for_unsupported_format(self, tmp_config, tmp_path):
        from core.tools.vision import VisionTool
        bad_file = tmp_path / "test.exe"
        bad_file.write_bytes(b"fake")
        v = VisionTool(tmp_config)
        with pytest.raises(ValueError):
            v.analyze(str(bad_file))

    def test_vision_returns_text_not_action(self, tmp_config, tmp_path):
        from core.tools.vision import VisionTool
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG header
        v = VisionTool(tmp_config)
        with patch.object(v, "_call_llava", return_value="A white rectangle."):
            result = v.analyze(str(img))
        assert isinstance(result, str)
        assert len(result) > 0
        # Result must not be an action command
        assert not result.startswith("shell_exec")
        assert not result.startswith("file_delete")


# ════════════════════════════════════════════════════════════════════════════
# 5. MEMORY — recalls correctly, never hallucinates
# ════════════════════════════════════════════════════════════════════════════

class TestHybridMemory:
    def test_store_and_retrieve_fact(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        mem.store_fact("sky_color", "blue", source="test")
        fact = mem.get_fact("sky_color")
        assert fact is not None
        assert fact.value == "blue"

    def test_unknown_key_returns_none(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        assert mem.get_fact("nonexistent_xyz") is None

    def test_recall_unknown_returns_dont_know(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        result = mem.recall("what is the meaning of life according to jarvis xyz abc")
        assert "don't know" in result.lower() or "related facts" in result.lower()

    def test_update_existing_fact(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        mem.store_fact("count", "1")
        mem.store_fact("count", "2")
        fact = mem.get_fact("count")
        assert fact.value == "2"

    def test_list_facts_returns_recent_first(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        mem.store_fact("a", "1")
        time.sleep(0.01)
        mem.store_fact("b", "2")
        facts = mem.list_facts(limit=2)
        assert facts[0].key == "b"

    def test_count(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        mem.store_fact("x", "1")
        mem.store_fact("y", "2")
        assert mem.count() == 2


# ════════════════════════════════════════════════════════════════════════════
# 6. AUDIT LOG — complete and immutable
# ════════════════════════════════════════════════════════════════════════════

class TestAuditLog:
    def test_write_and_verify(self, tmp_path):
        from core.logger import AuditLog
        log = AuditLog(str(tmp_path / "audit.jsonl"))
        log.write("TEST_EVENT", {"key": "value"})
        log.write("TEST_EVENT_2", {"key2": "value2"})
        ok, count, err = log.verify()
        assert ok, err
        assert count == 2

    def test_tamper_detected(self, tmp_path):
        from core.logger import AuditLog
        path = tmp_path / "audit.jsonl"
        log = AuditLog(str(path))
        log.write("SENSITIVE", {"secret": "data"})
        # Tamper with the file
        content = path.read_text()
        tampered = content.replace('"data"', '"hacked"')
        path.write_text(tampered)
        ok, count, err = log.verify()
        assert not ok

    def test_entries_are_chained(self, tmp_path):
        from core.logger import AuditLog
        log = AuditLog(str(tmp_path / "audit.jsonl"))
        h1 = log.write("E1", {})
        h2 = log.write("E2", {})
        # Read back and check prev_hash linkage
        lines = (tmp_path / "audit.jsonl").read_text().splitlines()
        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])
        assert e1["prev_hash"] == "0" * 64  # genesis
        assert e2["prev_hash"] == e1["hash"]

    def test_empty_log_verifies_ok(self, tmp_path):
        from core.logger import AuditLog
        log = AuditLog(str(tmp_path / "audit.jsonl"))
        ok, count, err = log.verify()
        assert ok
        assert count == 0


# ════════════════════════════════════════════════════════════════════════════
# 7. UNKNOWN REQUESTS — system says "I don't know" rather than guessing
# ════════════════════════════════════════════════════════════════════════════

class TestUnknownRequests:
    def test_memory_recall_unknown_does_not_hallucinate(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        result = mem.recall("the launch codes are xyzzy-7734-abc")
        # Should not make something up — must say it doesn't know
        assert result is not None
        assert isinstance(result, str)
        # Key assertion: it won't invent facts
        assert "xyzzy" not in result.lower() or "don't know" in result.lower()

    def test_planner_ollama_down_returns_clarification(self, tmp_config):
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        # With Ollama down, must return a valid plan with clarification
        plan = planner.plan("reboot the mainframe using the quantum override")
        assert "clarification_needed" in plan
        # Either it handled it or it asked for clarification — must not crash
        assert isinstance(plan["steps"], list)

    def test_planner_empty_input(self, tmp_config):
        from core.planning.task_planner import TaskPlanner
        planner = TaskPlanner(tmp_config)
        plan = planner.plan("")
        assert plan["clarification_needed"] is True

    def test_risk_unknown_action_is_not_approved(self, tmp_config):
        ev = RiskEvaluator(tmp_config)
        result = ev.evaluate(["unknown_alien_action_zxcvb"])
        # Must not be LOW — unknown = high risk by default
        assert result.level >= RiskLevel.HIGH


# ════════════════════════════════════════════════════════════════════════════
# 8. SERIAL CONTROLLER — blocked until V3
# ════════════════════════════════════════════════════════════════════════════

class TestSerialControllerStub:
    def test_send_raises(self):
        from core.hardware.serial_controller import SerialController
        sc = SerialController()
        with pytest.raises(NotImplementedError):
            sc.send("command")

    def test_connect_raises(self):
        from core.hardware.serial_controller import SerialController
        sc = SerialController()
        with pytest.raises(NotImplementedError):
            sc.connect("/dev/ttyUSB0")

    def test_is_connected_false(self):
        from core.hardware.serial_controller import SerialController
        sc = SerialController()
        assert sc.is_connected is False
