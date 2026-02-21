"""
tests/test_v3_execution.py - Basic V3 execution path checks.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from core.execution.dispatcher import ToolDispatcher
from core.hardware.serial_controller import SerialController
from core.memory.hybrid_memory import HybridMemory
from core.state_machine import State, StateMachine


@pytest.fixture
def v3_config(tmp_path):
    cfg = configparser.ConfigParser()
    data_dir = tmp_path / "data"
    cfg["memory"] = {
        "data_dir": str(data_dir),
        "sqlite_file": str(data_dir / "jarvis_memory.db"),
        "chroma_dir": str(data_dir / "chroma"),
        "embedding_model": "all-MiniLM-L6-v2",
        "semantic_top_k": "5",
    }
    cfg["execution"] = {
        "safe_directories": str(tmp_path / "sandbox"),
        "max_read_bytes": "50000",
        "allowed_apps": "notepad,calc",
    }
    cfg["hardware"] = {"enabled": "false"}
    return cfg


def test_fsm_v3_execution_transition_path():
    fsm = StateMachine()
    fsm.transition(State.PLANNING)
    fsm.transition(State.EXECUTING)
    fsm.transition(State.SPEAKING)
    fsm.transition(State.IDLE)
    assert fsm.state == State.IDLE


def test_dispatcher_blocks_outside_sandbox(v3_config):
    mem = HybridMemory(v3_config)
    dispatcher = ToolDispatcher(v3_config, memory=mem)
    plan = {
        "steps": [
            {
                "id": 1,
                "action": "file_read",
                "params": {"path": str(Path("C:/Windows/System32/drivers/etc/hosts"))},
            }
        ]
    }
    results = dispatcher.execute_plan(plan)
    assert results[0].success is False
    assert "outside safe directories" in (results[0].error or "")


def test_dispatcher_file_write_and_read(v3_config):
    mem = HybridMemory(v3_config)
    dispatcher = ToolDispatcher(v3_config, memory=mem)
    plan = {
        "steps": [
            {
                "id": 1,
                "action": "file_write",
                "params": {"path": "notes.txt", "content": "hello from v3"},
            },
            {"id": 2, "action": "file_read", "params": {"path": "notes.txt"}},
        ]
    }
    results = dispatcher.execute_plan(plan)
    assert results[0].success is True
    assert results[1].success is True
    assert "hello from v3" in results[1].output


def test_serial_controller_disabled_by_default(v3_config):
    serial = SerialController(config=v3_config)
    with pytest.raises(NotImplementedError):
        serial.send("LED_ON")
