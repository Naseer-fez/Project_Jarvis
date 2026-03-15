"""
tests/test_phase34.py - Phase 3/4 coverage:
- Risk-gated confirmation behavior
- Screen understanding and vision-guided click dispatcher actions
"""

from __future__ import annotations

import configparser

import pytest

from core.controller import Controller
from core.execution.dispatcher import ToolDispatcher
from core.memory.hybrid_memory import HybridMemory
from core.risk_evaluator import RiskLevel


class _DummyVision:
    def __init__(self, response: str):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def analyze(self, image_path: str, prompt: str = "Describe this image.") -> str:
        self.calls.append((image_path, prompt))
        return self._response


@pytest.fixture
def phase_cfg(tmp_path):
    cfg = configparser.ConfigParser()
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    sandbox = tmp_path / "sandbox"

    cfg["general"] = {"name": "Jarvis", "version": "2.0.0"}
    cfg["ollama"] = {
        "base_url": "http://localhost:11434",
        "planner_model": "deepseek-r1:8b",
        "vision_model": "llava",
        "request_timeout_s": "10",
    }
    cfg["memory"] = {
        "data_dir": str(data_dir),
        "sqlite_file": str(data_dir / "jarvis_memory.db"),
        "chroma_dir": str(data_dir / "chroma"),
        "embedding_model": "all-MiniLM-L6-v2",
        "semantic_top_k": "5",
    }
    cfg["risk"] = {
        "forbidden_actions": "shell_exec,file_delete",
        "high_risk_actions": "vision_click,gui_click,file_write",
        "medium_risk_actions": "screen_understand,web_search,file_read",
        "low_risk_actions": "memory_read,memory_write,system_stats,vision_analyze",
        "voice_confirm_threshold": "MEDIUM",
    }
    cfg["logging"] = {
        "log_dir": str(logs_dir),
        "audit_file": str(logs_dir / "audit.jsonl"),
        "app_file": str(logs_dir / "app.log"),
        "level": "INFO",
    }
    cfg["execution"] = {
        "safe_directories": str(sandbox),
        "max_read_bytes": "100000",
        "allowed_apps": "notepad",
        "allow_app_launch": "true",
        "allow_gui_automation": "true",
        "allow_web_search": "true",
    }
    cfg["hardware"] = {"enabled": "false"}
    cfg["voice"] = {"enabled": "false"}
    return cfg


@pytest.mark.skip(reason="Legacy _requires_confirmation logic removed in Controller V2")
def test_controller_confirmation_threshold_medium(phase_cfg):
    ctrl = Controller(phase_cfg, voice=False)

    class _Risk:
        level = RiskLevel.MEDIUM
        is_blocked = False

    assert ctrl._requires_confirmation(_Risk()) is True

    phase_cfg.set("risk", "voice_confirm_threshold", "HIGH")
    ctrl2 = Controller(phase_cfg, voice=False)
    assert ctrl2._requires_confirmation(_Risk()) is False


def test_dispatcher_screen_understand_action(phase_cfg, tmp_path, monkeypatch):
    mem = HybridMemory(phase_cfg)
    vision = _DummyVision("Desktop shows browser and terminal.")
    dispatcher = ToolDispatcher(phase_cfg, memory=mem, vision=vision)

    fake_image = tmp_path / "shot.png"
    fake_image.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        dispatcher,
        "_capture_screen_image",
        lambda output_path, capture_mode: (fake_image, (0, 0)),
    )

    results = dispatcher.execute_plan(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "screen_understand",
                    "params": {"capture_mode": "active_monitor"},
                }
            ]
        }
    )
    assert results[0].success is True
    assert "Desktop shows browser" in results[0].output
    assert vision.calls


def test_dispatcher_vision_click_dry_run(phase_cfg, tmp_path, monkeypatch):
    mem = HybridMemory(phase_cfg)
    vision = _DummyVision('{"x": 10, "y": 20, "confidence": 0.9, "reason": "button", "not_found": false}')
    dispatcher = ToolDispatcher(phase_cfg, memory=mem, vision=vision)

    fake_image = tmp_path / "shot2.png"
    fake_image.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        dispatcher,
        "_capture_screen_image",
        lambda output_path, capture_mode: (fake_image, (100, 50)),
    )

    clicked: list[dict] = []
    monkeypatch.setattr(dispatcher, "_gui_click", lambda params: clicked.append(params) or "ok")

    results = dispatcher.execute_plan(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "vision_click",
                    "params": {"target": "submit button", "dry_run": True},
                }
            ]
        }
    )
    assert results[0].success is True
    assert "screen=(110,70)" in results[0].output
    assert clicked == []


def test_dispatcher_vision_click_blocked_when_gui_disabled(phase_cfg):
    phase_cfg.set("execution", "allow_gui_automation", "false")
    mem = HybridMemory(phase_cfg)
    vision = _DummyVision('{"x": 1, "y": 2, "confidence": 1.0, "reason": "x", "not_found": false}')
    dispatcher = ToolDispatcher(phase_cfg, memory=mem, vision=vision)

    results = dispatcher.execute_plan(
        {"steps": [{"id": 1, "action": "vision_click", "params": {"target": "ok"}}]}
    )
    assert results[0].success is False
    assert "disabled by config" in (results[0].error or "")

