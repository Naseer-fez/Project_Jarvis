from __future__ import annotations

import asyncio
import configparser
import json
import time
from pathlib import Path

import pytest

from core.execution.async_task_manager import AsyncTaskManager
from core.execution.dispatcher import ToolDispatcher
from core.memory.hybrid_memory import HybridMemory
from core.permission_matrix import PermissionMatrix
from core.planning.task_planner import TaskPlanner
from core.risk_evaluator import RiskEvaluator, RiskLevel


@pytest.fixture
def v4_config(tmp_path):
    cfg = configparser.ConfigParser()
    data_dir = tmp_path / "data"
    sandbox = tmp_path / "sandbox"
    cfg["memory"] = {
        "data_dir": str(data_dir),
        "sqlite_file": str(data_dir / "jarvis_memory.db"),
        "chroma_dir": str(data_dir / "chroma"),
        "embedding_model": "all-MiniLM-L6-v2",
        "semantic_top_k": "5",
        "stale_action_days": "30",
        "decay_cleanup_on_start": "false",
    }
    cfg["execution"] = {
        "safe_directories": str(sandbox),
        "max_read_bytes": "50000",
        "allowed_apps": "notepad,calc",
        "sandboxed_execution": "true",
        "rollback_support": "true",
        "timeout_handling": "true",
        "step_timeout_s": "0.2",
        "stop_on_failure": "false",
        "rollback_on_failure": "true",
        "max_step_workers": "2",
    }
    cfg["risk"] = {
        "forbidden_actions": "shell_exec,file_delete",
        "critical_actions": "shell_exec,file_delete",
        "blocked_actions": "shell_exec,file_delete",
        "high_risk_actions": "file_write,vision_click",
        "medium_risk_actions": "file_read,web_search",
        "low_risk_actions": "memory_read,memory_write,speak,display",
        "user_confirmed_actions": "file_write,vision_click",
        "voice_confirm_threshold": "MEDIUM",
    }
    cfg["ollama"] = {
        "base_url": "http://localhost:11434",
        "planner_model": "deepseek-r1:8b",
        "request_timeout_s": "5",
    }
    cfg["concurrency"] = {"max_parallel_tasks": "2"}
    cfg["plugins"] = {"directory": str(tmp_path / "plugins"), "enabled_scopes": "core"}
    return cfg


def test_planner_schema_fields_present(v4_config):
    planner = TaskPlanner(v4_config)
    raw = json.dumps(
        {
            "intent": "write a note",
            "summary": "Write notes file.",
            "confidence": 0.8,
            "steps": [
                {
                    "id": 1,
                    "action": "file_write",
                    "description": "Write note",
                    "params": {"path": "notes.txt", "content": "hello"},
                }
            ],
        }
    )
    planner._call_ollama = lambda prompt: raw  # type: ignore[method-assign]
    plan = planner.plan("write a note")

    assert plan["intent"] == "write a note"
    assert "tools_required" in plan
    assert plan["tools_required"] == ["file_write"]
    assert "risk_level" in plan
    assert plan["risk_level"] in {"low", "medium", "high", "critical"}
    assert "confirmation_required" in plan


def test_permission_matrix_block_and_confirm(v4_config):
    matrix = PermissionMatrix(v4_config)
    result = matrix.evaluate(["file_write", "shell_exec", "memory_read"])
    assert "shell_exec" in result.blocked_actions
    assert "file_write" in result.confirmation_actions
    assert result.has_blocked
    assert result.needs_confirmation


def test_risk_evaluator_critical_level(v4_config):
    evaluator = RiskEvaluator(v4_config)
    res = evaluator.evaluate(["shell_exec"])
    assert res.is_blocked
    assert res.level >= RiskLevel.CRITICAL


def test_dispatcher_rollback_on_failure(v4_config):
    mem = HybridMemory(v4_config)
    dispatcher = ToolDispatcher(v4_config, memory=mem)
    safe_root = Path(v4_config["execution"]["safe_directories"])
    target = safe_root / "notes.txt"

    plan = {
        "steps": [
            {
                "id": 1,
                "action": "file_write",
                "params": {"path": "notes.txt", "content": "hello"},
            },
            {
                "id": 2,
                "action": "file_read",
                "params": {"path": "missing.txt"},
            },
        ]
    }
    results = dispatcher.execute_plan(plan)
    assert results[0].success is True
    assert results[1].success is False
    assert target.exists() is False


def test_dispatcher_timeout_handling(v4_config):
    mem = HybridMemory(v4_config)
    dispatcher = ToolDispatcher(v4_config, memory=mem)

    def _slow(params):
        del params
        time.sleep(0.6)
        return "done"

    dispatcher.register_action("slow_action", _slow)
    result = dispatcher.execute_plan(
        {"steps": [{"id": 1, "action": "slow_action", "params": {}}]}
    )[0]
    assert result.success is False
    assert result.timed_out is True


def test_memory_expansion_and_decay(v4_config):
    mem = HybridMemory(v4_config)
    mem.store_action("file_write", "ok", success=True, metadata={"k": "v"})
    mem.store_failure("web_search", "timeout", metadata={"q": "x"})
    mem.set_preference("voice_tone", "calm", category="voice_setting")

    prefs = mem.get_preferences("voice_setting")
    assert prefs.get("voice_tone") == "calm"
    actions = mem.recent_actions(limit=10)
    assert len(actions) >= 2
    assert any(a["success"] is False for a in actions)

    cleanup = mem.cleanup_stale_data(max_age_days=0)
    assert cleanup["episodic_removed"] >= 1


def test_async_task_manager_priority_and_cancel():
    async def _run():
        manager = AsyncTaskManager(max_parallel=1)
        order: list[str] = []

        async def _job(name: str):
            order.append(name)
            await asyncio.sleep(0.01)
            return name

        low_id, low_future = await manager.submit(lambda: _job("low"), priority=10, name="low")
        high_id, high_future = await manager.submit(lambda: _job("high"), priority=1, name="high")
        cancel_id, cancel_future = await manager.submit(
            lambda: _job("cancel"), priority=20, name="cancel"
        )
        assert low_id != high_id
        assert manager.cancel(cancel_id) is True
        assert cancel_future.cancelled()

        await manager.start()
        high_result = await high_future
        low_result = await low_future
        await manager.stop()

        assert high_result == "high"
        assert low_result == "low"
        assert order[0] == "high"

    asyncio.run(_run())
