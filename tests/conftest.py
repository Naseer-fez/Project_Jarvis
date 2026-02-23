"""
tests/conftest.py — shared pytest fixtures for Jarvis V2.
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

import pytest

# Ensure project root is on path regardless of where pytest is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def minimal_config() -> configparser.ConfigParser:
    """A bare-minimum ConfigParser with no sections — safe default."""
    return configparser.ConfigParser()


@pytest.fixture()
def full_config() -> configparser.ConfigParser:
    """A realistic ConfigParser with common sections pre-populated."""
    config = configparser.ConfigParser()
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