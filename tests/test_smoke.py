"""
tests/test_smoke.py — Smoke tests for Jarvis V2 entry point.

These tests verify the entry point behaves correctly without
requiring a real Controller, real config, or real audit log.

Run:
  pytest tests/test_smoke.py -v
  pytest tests/test_smoke.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import configparser
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Make project root importable regardless of where pytest runs ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import main as jarvis_main
from main import (
    ExitCode,
    _ShutdownCoordinator,
    apply_cli_overrides,
    async_main,
    load_config,
    parse_args,
)


# ─────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────
def _make_args(**kwargs) -> Any:
    """Return a minimal Namespace mimicking parse_args() output."""
    import argparse

    defaults = dict(
        voice=False,
        verify=False,
        config="config/jarvis.ini",
        log_level=None,
        session_name=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_logger_mod(
    *,
    audit_ok: bool = True,
    audit_count: int = 42,
    audit_err: str | None = None,
) -> ModuleType:
    """
    Return a fake core.logging.logger module.
    setup() and get() succeed; verify_audit() returns configurable results.
    """
    mod = MagicMock()
    mod.setup = MagicMock()
    mod.get = MagicMock(return_value=MagicMock())
    mod.verify_audit = MagicMock(return_value=(audit_ok, audit_count, audit_err))
    return mod


def _make_controller(*, has_run_cli: bool = True, run_cli_raises=None) -> MagicMock:
    ctrl = MagicMock()
    ctrl.start = AsyncMock()
    ctrl.shutdown = AsyncMock()

    if has_run_cli:
        if run_cli_raises:
            ctrl.run_cli = AsyncMock(side_effect=run_cli_raises)
        else:
            ctrl.run_cli = AsyncMock(return_value=None)
    else:
        del ctrl.run_cli  # simulate missing attribute

    return ctrl


# ─────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────
class TestLoadConfig:
    def test_returns_empty_config_when_file_missing_in_dev(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("JARVIS_ENV", "development")
        config = load_config(str(tmp_path / "nonexistent.ini"))
        assert isinstance(config, configparser.ConfigParser)

    def test_exits_config_error_when_file_missing_in_prod(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("JARVIS_ENV", "production")
        with pytest.raises(SystemExit) as exc_info:
            load_config(str(tmp_path / "nonexistent.ini"))
        assert exc_info.value.code == ExitCode.CONFIG_ERROR

    def test_loads_valid_ini(self, tmp_path):
        ini = tmp_path / "jarvis.ini"
        ini.write_text("[general]\nname = test\n", encoding="utf-8")
        config = load_config(str(ini))
        assert config.get("general", "name") == "test"

    def test_exits_on_malformed_ini(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        bad = tmp_path / "bad.ini"
        # write bytes that confuse the parser
        bad.write_text("[broken\nno_closing_bracket\n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            load_config(str(bad))
        assert exc_info.value.code == ExitCode.CONFIG_ERROR


# ─────────────────────────────────────────────────────────────
# apply_cli_overrides
# ─────────────────────────────────────────────────────────────
class TestApplyCliOverrides:
    def test_applies_log_level(self):
        config = configparser.ConfigParser()
        apply_cli_overrides(config, _make_args(log_level="DEBUG"))
        assert config["logging"]["level"] == "DEBUG"

    def test_applies_session_name(self):
        config = configparser.ConfigParser()
        apply_cli_overrides(config, _make_args(session_name="my-session"))
        assert config["general"]["session_name"] == "my-session"

    def test_does_not_create_sections_when_nothing_to_override(self):
        config = configparser.ConfigParser()
        apply_cli_overrides(config, _make_args())
        assert "logging" not in config
        assert "general" not in config

    def test_does_not_clobber_existing_keys(self):
        config = configparser.ConfigParser()
        config["logging"] = {"level": "INFO", "file": "/var/log/jarvis.log"}
        apply_cli_overrides(config, _make_args(log_level="DEBUG"))
        # level overridden, file preserved
        assert config["logging"]["level"] == "DEBUG"
        assert config["logging"]["file"] == "/var/log/jarvis.log"


# ─────────────────────────────────────────────────────────────
# parse_args
# ─────────────────────────────────────────────────────────────
class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("JARVIS_CONFIG", raising=False)
        monkeypatch.delenv("JARVIS_LOG_LEVEL", raising=False)
        monkeypatch.setattr(sys, "argv", ["main.py"])
        args = parse_args()
        assert args.voice is False
        assert args.verify is False
        assert args.log_level is None
        assert args.session_name is None

    def test_voice_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["main.py", "--voice"])
        args = parse_args()
        assert args.voice is True

    def test_verify_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["main.py", "--verify"])
        args = parse_args()
        assert args.verify is True

    def test_env_var_config(self, monkeypatch):
        monkeypatch.setenv("JARVIS_CONFIG", "/custom/path.ini")
        monkeypatch.setattr(sys, "argv", ["main.py"])
        args = parse_args()
        assert args.config == "/custom/path.ini"

    def test_env_var_log_level(self, monkeypatch):
        monkeypatch.setenv("JARVIS_LOG_LEVEL", "WARNING")
        monkeypatch.setattr(sys, "argv", ["main.py"])
        args = parse_args()
        assert args.log_level == "WARNING"

    def test_cli_log_level_overrides_env(self, monkeypatch):
        monkeypatch.setenv("JARVIS_LOG_LEVEL", "WARNING")
        monkeypatch.setattr(sys, "argv", ["main.py", "--log-level", "DEBUG"])
        args = parse_args()
        assert args.log_level == "DEBUG"


# ─────────────────────────────────────────────────────────────
# _ShutdownCoordinator
# ─────────────────────────────────────────────────────────────
class TestShutdownCoordinator:
    def test_wait_resolves_after_request(self):
        async def _run():
            loop = asyncio.get_running_loop()
            coord = _ShutdownCoordinator(loop)
            coord.request_shutdown("test")
            # Should return immediately since event is already set
            await asyncio.wait_for(coord.wait(), timeout=1.0)

        asyncio.run(_run())

    def test_wait_blocks_until_requested(self):
        async def _run():
            loop = asyncio.get_running_loop()
            coord = _ShutdownCoordinator(loop)

            async def _trigger():
                await asyncio.sleep(0.05)
                coord.request_shutdown("delayed")

            trigger_task = asyncio.create_task(_trigger())
            await asyncio.wait_for(coord.wait(), timeout=2.0)
            await trigger_task

        asyncio.run(_run())

    def test_multiple_requests_are_idempotent(self):
        async def _run():
            loop = asyncio.get_running_loop()
            coord = _ShutdownCoordinator(loop)
            coord.request_shutdown("first")
            coord.request_shutdown("second")  # should not raise
            await asyncio.wait_for(coord.wait(), timeout=1.0)

        asyncio.run(_run())


# ─────────────────────────────────────────────────────────────
# async_main — audit path
# ─────────────────────────────────────────────────────────────
class TestAsyncMainAudit:
    @pytest.fixture(autouse=True)
    def _patch_logging(self):
        """Always patch core.logging so tests don't need real files."""
        logger_mod = _make_logger_mod()
        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.logging": MagicMock(), "core.logging.logger": logger_mod},
        ):
            with patch("main.async_main.__globals__", {**jarvis_main.async_main.__globals__}):
                pass
            yield logger_mod

    def _run(self, args):
        return asyncio.run(async_main(args))

    def test_verify_ok_returns_exit_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        logger_mod = _make_logger_mod(audit_ok=True, audit_count=7)

        with patch("builtins.__import__", side_effect=_make_import_hook(logger_mod)):
            code = asyncio.run(async_main(_make_args(verify=True, config=str(ini))))

        assert code == ExitCode.OK

    def test_verify_fail_returns_audit_failed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        logger_mod = _make_logger_mod(audit_ok=False, audit_count=3, audit_err="hash mismatch")

        with patch("builtins.__import__", side_effect=_make_import_hook(logger_mod)):
            code = asyncio.run(async_main(_make_args(verify=True, config=str(ini))))

        assert code == ExitCode.AUDIT_FAILED


def _make_import_hook(logger_mod):
    """
    Intercept imports of core.logging.logger and return our fake module.
    All other imports pass through normally.
    """
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _hook(name, *args, **kwargs):
        if name in ("core.logging", "core.logging.logger"):
            return MagicMock(logger=logger_mod)
        return real_import(name, *args, **kwargs)

    return _hook


# ─────────────────────────────────────────────────────────────
# async_main — controller lifecycle
# ─────────────────────────────────────────────────────────────
class TestAsyncMainController:
    """
    Test controller startup, run, and shutdown lifecycle using
    full patching of core.logging and core.controller.
    """

    def _build_patches(self, controller, logger_mod):
        return {
            "core.logging.logger": logger_mod,
            "core.controller.Controller": MagicMock(return_value=controller),
        }

    def _run_with_patches(self, args, controller, logger_mod):
        patches = self._build_patches(controller, logger_mod)

        with patch("main.async_main", wraps=async_main):
            # Patch the imports inside async_main at the module level
            with patch.dict(sys.modules, {
                "core": MagicMock(),
                "core.logging": MagicMock(logger=logger_mod),
                "core.controller": MagicMock(Controller=MagicMock(return_value=controller)),
            }):
                return asyncio.run(async_main(args))

    def test_clean_run_returns_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        controller = _make_controller()
        logger_mod = _make_logger_mod()

        code = self._run_with_patches(
            _make_args(config=str(ini)), controller, logger_mod
        )
        assert code == ExitCode.OK
        controller.start.assert_awaited_once()
        controller.shutdown.assert_awaited_once()

    def test_startup_exception_returns_startup_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        controller = _make_controller()
        controller.start = AsyncMock(side_effect=RuntimeError("boom"))
        logger_mod = _make_logger_mod()

        code = self._run_with_patches(
            _make_args(config=str(ini)), controller, logger_mod
        )
        assert code == ExitCode.STARTUP_ERROR

    def test_run_cli_exception_returns_generic_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        controller = _make_controller(run_cli_raises=RuntimeError("cli crashed"))
        logger_mod = _make_logger_mod()

        code = self._run_with_patches(
            _make_args(config=str(ini)), controller, logger_mod
        )
        assert code == ExitCode.GENERIC_ERROR
        controller.shutdown.assert_awaited_once()

    def test_shutdown_always_called_even_on_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        controller = _make_controller(run_cli_raises=ValueError("unexpected"))
        logger_mod = _make_logger_mod()

        self._run_with_patches(_make_args(config=str(ini)), controller, logger_mod)
        controller.shutdown.assert_awaited_once()

    def test_headless_mode_exits_cleanly_on_signal(self, tmp_path, monkeypatch):
        """Controller without run_cli() should wait for shutdown signal."""
        monkeypatch.setenv("JARVIS_ENV", "development")
        ini = tmp_path / "j.ini"
        ini.write_text("", encoding="utf-8")

        controller = _make_controller(has_run_cli=False)
        logger_mod = _make_logger_mod()

        async def _patched_main():
            # Patch shutdown coordinator to fire immediately
            original = jarvis_main._ShutdownCoordinator

            class _ImmediateShutdown(original):
                async def wait(self):
                    await asyncio.sleep(0)  # yield, then return immediately

            with patch("main._ShutdownCoordinator", _ImmediateShutdown):
                with patch.dict(sys.modules, {
                    "core": MagicMock(),
                    "core.logging": MagicMock(logger=logger_mod),
                    "core.controller": MagicMock(
                        Controller=MagicMock(return_value=controller)
                    ),
                }):
                    return await async_main(_make_args(config=str(ini)))

        code = asyncio.run(_patched_main())
        assert code == ExitCode.OK
        controller.shutdown.assert_awaited_once()


# ─────────────────────────────────────────────────────────────
# ExitCode constants
# ─────────────────────────────────────────────────────────────
class TestExitCodes:
    def test_all_codes_are_unique(self):
        codes = [
            ExitCode.OK,
            ExitCode.GENERIC_ERROR,
            ExitCode.CONFIG_ERROR,
            ExitCode.AUDIT_FAILED,
            ExitCode.STARTUP_ERROR,
        ]
        assert len(codes) == len(set(codes)), "Exit codes must be unique"

    def test_ok_is_zero(self):
        assert ExitCode.OK == 0

    def test_all_errors_are_nonzero(self):
        assert ExitCode.GENERIC_ERROR != 0
        assert ExitCode.CONFIG_ERROR != 0
        assert ExitCode.AUDIT_FAILED != 0
