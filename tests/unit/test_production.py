from __future__ import annotations

import os
from configparser import ConfigParser
from unittest import mock

from core.ops.production import is_production, validate_production_config


def test_is_production():
    # Test fallback to config
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "production")

    with mock.patch.dict(os.environ, {}, clear=True):
        assert is_production(config) is True

    config.set("general", "environment", "development")
    with mock.patch.dict(os.environ, {}, clear=True):
        assert is_production(config) is False

    # Test override by environment variable
    with mock.patch.dict(os.environ, {"JARVIS_ENV": "production"}):
        assert is_production(config) is True

    with mock.patch.dict(os.environ, {"JARVIS_ENV": "development"}):
        assert is_production(config) is False


def test_validation_in_development():
    # Dev settings don't require production secrets
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "development")

    with mock.patch.dict(os.environ, {}, clear=True):
        check = validate_production_config(config)
        assert check.ok is True
        assert len(check.errors) == 0


def test_production_secret_keys():
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "production")

    # Insecure secret key in production
    env_vars = {
        "JARVIS_ENV": "production",
        "JARVIS_SECRET_KEY": "jarvis",
        "JARVIS_ADMIN_USER": "admin",
        "JARVIS_ADMIN_PASSWORD": "securepassword123",
    }
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is False
        assert any("JARVIS_SECRET_KEY" in err for err in check.errors)

    # Empty secret key in production
    env_vars["JARVIS_SECRET_KEY"] = ""
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is False
        assert any("JARVIS_SECRET_KEY" in err for err in check.errors)


def test_production_admin_credentials():
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "production")

    # Missing admin credentials in production
    env_vars = {
        "JARVIS_ENV": "production",
        "JARVIS_SECRET_KEY": "super-secret-key-12345",
    }
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is False
        assert any("JARVIS_ADMIN_USER" in err for err in check.errors)

    # Short admin password in production
    env_vars["JARVIS_ADMIN_USER"] = "admin"
    env_vars["JARVIS_ADMIN_PASSWORD"] = "short"
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is False
        assert any("JARVIS_ADMIN_PASSWORD must be at least 12 characters" in err for err in check.errors)


def test_production_dashboard_bindings():
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "production")
    config.add_section("dashboard")
    config.set("dashboard", "host", "0.0.0.0")

    env_vars = {
        "JARVIS_ENV": "production",
        "JARVIS_SECRET_KEY": "super-secret-key-12345",
        "JARVIS_ADMIN_USER": "admin",
        "JARVIS_ADMIN_PASSWORD": "securepassword123",
        "JARVIS_REQUIRE_HTTPS": "true",
    }

    # Public dashboard bind in production without reverse proxy ACK
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config, dashboard_enabled=True)
        assert check.ok is False
        assert any("Public dashboard binding" in err for err in check.errors)

    # With ACK, validation should succeed (or at least pass the dashboard check)
    env_vars["JARVIS_PUBLIC_DASHBOARD_ACK"] = "true"
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config, dashboard_enabled=True)
        assert check.ok is True


def test_production_risky_execution_flags():
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "production")
    config.add_section("execution")
    config.set("execution", "allow_gui_automation", "true")
    config.set("execution", "allow_shell_execution", "true")
    config.add_section("hardware")
    config.set("hardware", "enabled", "true")

    env_vars = {
        "JARVIS_ENV": "production",
        "JARVIS_SECRET_KEY": "super-secret-key-12345",
        "JARVIS_ADMIN_USER": "admin",
        "JARVIS_ADMIN_PASSWORD": "securepassword123",
    }

    # Allowed in config but no env confirmation flags set
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is False
        assert any("allow_gui_automation" in err for err in check.errors)
        assert any("allow_shell_execution" in err for err in check.errors)
        assert any("hardware_enabled" in err for err in check.errors)

    # Set confirmation flags
    env_vars.update({
        "JARVIS_ENABLE_GUI_AUTOMATION": "true",
        "JARVIS_ENABLE_SHELL": "true",
        "JARVIS_ENABLE_HARDWARE": "true",
    })
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is True


def test_production_model_providers():
    config = ConfigParser()
    config.add_section("general")
    config.set("general", "environment", "production")

    env_vars = {
        "JARVIS_ENV": "production",
        "JARVIS_SECRET_KEY": "super-secret-key-12345",
        "JARVIS_ADMIN_USER": "admin",
        "JARVIS_ADMIN_PASSWORD": "securepassword123",
        "JARVIS_MODEL_PROVIDER_ORDER": "",
    }

    # Empty provider list
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is False
        assert any("At least one model provider must be configured" in err for err in check.errors)

    # Provider list without ollama (generates warning)
    env_vars["JARVIS_MODEL_PROVIDER_ORDER"] = "gemini,openai"
    with mock.patch.dict(os.environ, env_vars, clear=True):
        check = validate_production_config(config)
        assert check.ok is True
        assert len(check.warnings) == 1
        assert "Ollama is not in JARVIS_MODEL_PROVIDER_ORDER" in check.warnings[0]
