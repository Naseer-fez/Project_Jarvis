"""
tests/conftest.py — shared pytest fixtures for Jarvis Session 8.
"""

from __future__ import annotations

import sys
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path regardless of where pytest is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def mock_config():
    cfg = ConfigParser()
    cfg["agent"] = {"max_iterations": "10"}
    cfg["models"] = {"chat_model": "mistral:7b", "fallback_model": "mistral:7b"}
    cfg["proactive"] = {"cpu_alert_threshold": "90"}
    return cfg


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.complete = MagicMock(
        return_value='{"communication_style": {"value": "casual", "confidence": 0.8}}'
    )
    llm.complete_json = MagicMock(return_value={})
    return llm


@pytest.fixture()
def mock_controller():
    ctrl = MagicMock()
    ctrl.process = MagicMock(return_value="test response")
    ctrl.session_id = "test-session"
    return ctrl


# ── Backward-compat aliases used by older test files ──────────────────────────

@pytest.fixture()
def minimal_config() -> ConfigParser:
    """A bare-minimum ConfigParser with no sections — safe default."""
    return ConfigParser()


@pytest.fixture()
def full_config() -> ConfigParser:
    """A realistic ConfigParser with common sections pre-populated."""
    config = ConfigParser()
    config["general"] = {"session_name": "test-session"}
    config["logging"] = {"level": "DEBUG", "file": "/tmp/jarvis-test.log"}
    config["voice"] = {"enabled": "false"}
    return config


@pytest.fixture()
def tmp_ini(tmp_path) -> Path:
    """An empty but valid INI file on disk."""
    ini = tmp_path / "jarvis.ini"
    ini.write_text("", encoding="utf-8")
    return ini
