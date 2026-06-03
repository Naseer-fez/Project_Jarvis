"""
tests/conftest.py — shared pytest fixtures for Jarvis.
Modernized to remove side-effects where possible and provide clean core mocks.
"""

from __future__ import annotations

import os
# We set these environment variables BEFORE importing application code to ensure
# offline mode and mock embeddings are strictly enforced.
os.environ["JARVIS_MOCK_EMBEDDINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path regardless of where pytest is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import threading  # noqa: E402

_current_test_tmp_path = threading.local()

@pytest.fixture(autouse=True)
def set_current_tmp_path(tmp_path):
    _current_test_tmp_path.value = tmp_path
    yield
    _current_test_tmp_path.value = None

# Wrap build_controller_services to isolate sqlite/chromadb files during tests
import core.controller.services  # noqa: E402
original_build = core.controller.services.build_controller_services

def wrapped_build(config, *args, **kwargs):
    tmp_path = getattr(_current_test_tmp_path, "value", None)
    if tmp_path is not None:
        if "db_path" not in kwargs or kwargs["db_path"] == "memory/memory.db":
            kwargs["db_path"] = str(tmp_path / "test_memory.db")
        if "chroma_path" not in kwargs or kwargs["chroma_path"] == "data/chroma":
            kwargs["chroma_path"] = str(tmp_path / "test_chroma")
            
        try:
            if not config.has_section("memory"):
                config.add_section("memory")
            if not config.has_option("memory", "db_path"):
                config.set("memory", "db_path", str(tmp_path / "test_memory.db"))
            if not config.has_option("memory", "chroma_path"):
                config.set("memory", "chroma_path", str(tmp_path / "test_chroma"))
        except (AttributeError, TypeError):
            # Fallback if config is a dictionary or custom object without ConfigParser interface
            if hasattr(config, "__setitem__"):
                if "memory" not in config:
                    config["memory"] = {}
                if isinstance(config["memory"], dict):
                    if "db_path" not in config["memory"]:
                        config["memory"]["db_path"] = str(tmp_path / "test_memory.db")
                    if "chroma_path" not in config["memory"]:
                        config["memory"]["chroma_path"] = str(tmp_path / "test_chroma")
    return original_build(config, *args, **kwargs)

core.controller.services.build_controller_services = wrapped_build


# Wrap AuthManager.__init__ to isolate the auth database file during tests
import core.security.auth  # noqa: E402
original_auth_manager_init = core.security.auth.AuthManager.__init__

def wrapped_auth_manager_init(self, db_path, *args, **kwargs):
    tmp_path = getattr(_current_test_tmp_path, "value", None)
    if tmp_path is not None:
        db_path = tmp_path / "test_auth.db"
    original_auth_manager_init(self, db_path, *args, **kwargs)

core.security.auth.AuthManager.__init__ = wrapped_auth_manager_init




@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Returns a temporary directory path."""
    return tmp_path


@pytest.fixture()
def mock_config() -> ConfigParser:
    """Returns a realistic ConfigParser with common sections pre-populated."""
    cfg = ConfigParser()
    cfg["general"] = {"session_name": "test-session"}
    cfg["agent"] = {"max_iterations": "10"}
    cfg["models"] = {"chat_model": "mistral:7b", "fallback_model": "mistral:7b"}
    cfg["proactive"] = {"cpu_alert_threshold": "90"}
    cfg["logging"] = {"level": "DEBUG", "file": "/tmp/jarvis-test.log"}
    cfg["voice"] = {"enabled": "false"}
    return cfg


@pytest.fixture()
def minimal_config() -> ConfigParser:
    """A bare-minimum ConfigParser with no sections — safe default."""
    return ConfigParser()


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Returns a mock LLM that simulates common responses."""
    llm = MagicMock()
    llm.complete = MagicMock(
        return_value='{"communication_style": {"value": "casual", "confidence": 0.8}}'
    )
    llm.complete_json = MagicMock(return_value={})
    return llm


@pytest.fixture()
def mock_controller() -> MagicMock:
    """Returns a mock Jarvis controller."""
    ctrl = MagicMock()
    ctrl.process = MagicMock(return_value="test response")
    ctrl.session_id = "test-session"
    return ctrl


@pytest.fixture()
def tmp_ini(tmp_path: Path) -> Path:
    """An empty but valid INI file on disk."""
    ini = tmp_path / "jarvis.ini"
    ini.write_text("", encoding="utf-8")
    return ini


# def pytest_sessionfinish(session, exitstatus):
#     """Force exit the process to prevent hangs from non-daemon threads in third-party libraries."""
#     import os
#     os._exit(exitstatus)


