"""
tests/test_risk_evaluator.py — Tests for RiskEvaluator and RiskLevel.
All external dependencies are mocked; tests run fully offline.
"""

import pytest
from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel, RiskResult


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def evaluator():
    return RiskEvaluator()


# ── Risk ordering ─────────────────────────────────────────────────────────────

def test_risk_level_ordering():
    """LOW < CONFIRM < HIGH < CRITICAL."""
    assert RiskLevel.LOW < RiskLevel.CONFIRM
    assert RiskLevel.CONFIRM < RiskLevel.HIGH
    assert RiskLevel.HIGH < RiskLevel.CRITICAL


# ── Single-tool evaluations ───────────────────────────────────────────────────

def test_read_file_is_low(evaluator):
    result = evaluator.evaluate(["read_file"])
    assert result.level == RiskLevel.LOW


def test_write_file_requires_confirm(evaluator):
    result = evaluator.evaluate(["write_file"])
    assert result.level == RiskLevel.CONFIRM
    assert result.requires_confirmation is True
    assert result.is_blocked is False


def test_delete_file_is_critical(evaluator):
    result = evaluator.evaluate(["delete_file"])
    assert result.level == RiskLevel.CRITICAL
    assert result.is_blocked is True


def test_unknown_tool_defaults_to_high(evaluator):
    result = evaluator.evaluate(["some_totally_unknown_tool_xyz"])
    assert result.level >= RiskLevel.HIGH
    assert "some_totally_unknown_tool_xyz" in result.high_risk_actions


# ── evaluate_plan() ───────────────────────────────────────────────────────────

def test_evaluate_plan_returns_highest_risk(evaluator):
    plan = {
        "steps": [
            {"action": "read_file"},
            {"action": "write_file"},
            {"action": "delete_file"},
        ]
    }
    result = evaluator.evaluate_plan(plan)
    # CRITICAL from delete_file should dominate
    assert result.level == RiskLevel.CRITICAL


def test_evaluate_plan_all_low(evaluator):
    plan = {
        "steps": [
            {"action": "read_file"},
            {"action": "list_directory"},
        ]
    }
    result = evaluator.evaluate_plan(plan)
    assert result.level == RiskLevel.LOW


def test_evaluate_plan_empty_steps(evaluator):
    result = evaluator.evaluate_plan({"steps": []})
    # Empty plan — should be LOW with no error
    assert result.level == RiskLevel.LOW


def test_evaluate_plan_no_steps_key(evaluator):
    result = evaluator.evaluate_plan({})
    assert result.level == RiskLevel.LOW


# ── Empty action list ─────────────────────────────────────────────────────────

def test_empty_action_list_returns_low(evaluator):
    result = evaluator.evaluate([])
    assert result.level == RiskLevel.LOW
    assert result.is_blocked is False


# ── Mixed risk plan ───────────────────────────────────────────────────────────

def test_mixed_actions_highest_wins(evaluator):
    """Mixing LOW + CONFIRM actions should return CONFIRM (no CRITICAL present)."""
    result = evaluator.evaluate(["read_file", "write_file", "list_directory"])
    assert result.level == RiskLevel.CONFIRM


# ── is_blocked / requires_confirmation ───────────────────────────────────────

def test_low_not_blocked(evaluator):
    result = evaluator.evaluate(["read_file"])
    assert not result.is_blocked
    assert not result.requires_confirmation


def test_confirm_requires_confirmation_not_blocked(evaluator):
    result = evaluator.evaluate(["write_file"])
    assert not result.is_blocked
    assert result.requires_confirmation


def test_critical_is_blocked(evaluator):
    result = evaluator.evaluate(["shell_exec"])
    assert result.is_blocked
