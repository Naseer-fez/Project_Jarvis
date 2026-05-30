from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


PUBLIC_HOSTS = {"0.0.0.0", "::", "[::]"}
DANGEROUS_ENV_FLAGS = {
    "allow_gui_automation": "JARVIS_ENABLE_GUI_AUTOMATION",
    "allow_shell_execution": "JARVIS_ENABLE_SHELL",
    "hardware_enabled": "JARVIS_ENABLE_HARDWARE",
}


@dataclass
class ProductionCheck:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _get(config: Any, section: str, key: str, fallback: str = "") -> str:
    try:
        return str(config.get(section, key, fallback=fallback))
    except Exception:
        return fallback


def _get_bool(config: Any, section: str, key: str, fallback: bool = False) -> bool:
    try:
        return bool(config.getboolean(section, key, fallback=fallback))
    except Exception:
        return fallback


def is_production(config: Any) -> bool:
    env = os.environ.get("JARVIS_ENV") or _get(config, "general", "environment", "")
    return env.strip().lower() == "production"


def validate_production_config(config: Any, *, dashboard_enabled: bool = False) -> ProductionCheck:
    result = ProductionCheck()
    prod = is_production(config)

    secret = os.environ.get("JARVIS_SECRET_KEY", "")
    if prod and (not secret or secret in {"jarvis", "development-only-secret"}):
        result.errors.append("JARVIS_SECRET_KEY must be set to a strong non-default value in production.")

    admin_user = os.environ.get("JARVIS_ADMIN_USER", "")
    admin_password = os.environ.get("JARVIS_ADMIN_PASSWORD", "")
    if prod and (not admin_user or not admin_password):
        result.errors.append("JARVIS_ADMIN_USER and JARVIS_ADMIN_PASSWORD are required for first production bootstrap.")
    if prod and admin_password and len(admin_password) < 12:
        result.errors.append("JARVIS_ADMIN_PASSWORD must be at least 12 characters.")

    if dashboard_enabled:
        host = os.environ.get("JARVIS_DASHBOARD_HOST") or _get(config, "dashboard", "host", "127.0.0.1")
        public_bind = host.strip() in PUBLIC_HOSTS
        require_https = os.environ.get("JARVIS_REQUIRE_HTTPS", "true").lower() != "false"
        proxy_ack = os.environ.get("JARVIS_PUBLIC_DASHBOARD_ACK", "").lower() == "true"
        if prod and public_bind and require_https and not proxy_ack:
            result.errors.append(
                "Public dashboard binding in production requires HTTPS/reverse-proxy acknowledgement via JARVIS_PUBLIC_DASHBOARD_ACK=true."
            )

    risky_enabled = {
        "allow_gui_automation": _get_bool(config, "execution", "allow_gui_automation", False),
        "allow_shell_execution": _get_bool(config, "execution", "allow_shell_execution", False),
        "hardware_enabled": _get_bool(config, "hardware", "enabled", False),
    }
    if prod:
        for key, enabled in risky_enabled.items():
            flag = DANGEROUS_ENV_FLAGS[key]
            if enabled and os.environ.get(flag, "").lower() != "true":
                result.errors.append(f"{key} is enabled but {flag}=true was not set.")

    provider_order = os.environ.get("JARVIS_MODEL_PROVIDER_ORDER") or _get(
        config,
        "models",
        "provider_order",
        "gemini,openai,groq,anthropic,ollama",
    )
    providers = [item.strip() for item in provider_order.split(",") if item.strip()]
    if prod and not providers:
        result.errors.append("At least one model provider must be configured.")
    if prod and "ollama" not in providers:
        result.warnings.append("Ollama is not in JARVIS_MODEL_PROVIDER_ORDER; local fallback is disabled.")

    return result


__all__ = ["ProductionCheck", "is_production", "validate_production_config"]
