"""
tests/test_phase5_serial.py - Phase 5 serial/hardware coverage.
"""

from __future__ import annotations

import configparser

import pytest

from core.execution.dispatcher import ToolDispatcher
from core.hardware.serial_controller import SerialController
from core.memory.hybrid_memory import HybridMemory


@pytest.fixture
def serial_cfg(tmp_path):
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
        "allowed_apps": "notepad",
        "allow_gui_automation": "false",
        "allow_web_search": "true",
    }
    cfg["hardware"] = {
        "enabled": "true",
        "default_port": "COM7",
        "baud_rate": "115200",
    }
    return cfg


def test_serial_controller_uses_default_port(serial_cfg, monkeypatch):
    created = {}

    class _FakeSerial:
        def __init__(self, port, baudrate, timeout):
            created["port"] = port
            created["baudrate"] = baudrate
            created["timeout"] = timeout
            self.is_open = True

        def close(self):
            self.is_open = False

    import serial

    monkeypatch.setattr(serial, "Serial", _FakeSerial)

    sc = SerialController(config=serial_cfg)
    sc.connect(port=None, baud_rate=None)
    assert created["port"] == "COM7"
    assert created["baudrate"] == 115200
    assert sc.is_connected


def test_dispatcher_physical_actuate_builds_command(serial_cfg):
    class _FakeSerialCtrl:
        def __init__(self):
            self.commands: list[str] = []

        def send(self, command: str):
            self.commands.append(command)
            return "ACK"

    fake = _FakeSerialCtrl()
    mem = HybridMemory(serial_cfg)
    dispatcher = ToolDispatcher(serial_cfg, memory=mem, serial_controller=fake)

    results = dispatcher.execute_plan(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "physical_actuate",
                    "params": {"device": "LIGHT", "state": "ON"},
                }
            ]
        }
    )
    assert results[0].success is True
    assert fake.commands == ["LIGHT:ON"]


def test_dispatcher_sensor_read_builds_read_command(serial_cfg):
    class _FakeSerialCtrl:
        def __init__(self):
            self.commands: list[str] = []

        def send(self, command: str):
            self.commands.append(command)
            return "24.6C"

    fake = _FakeSerialCtrl()
    mem = HybridMemory(serial_cfg)
    dispatcher = ToolDispatcher(serial_cfg, memory=mem, serial_controller=fake)

    results = dispatcher.execute_plan(
        {"steps": [{"id": 1, "action": "sensor_read", "params": {"sensor": "temp"}}]}
    )
    assert results[0].success is True
    assert fake.commands == ["READ:TEMP"]
    assert "24.6C" in results[0].output
