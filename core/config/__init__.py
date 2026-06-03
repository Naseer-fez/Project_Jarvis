"""Typed, unified config manager for Project Jarvis."""

from __future__ import annotations

import configparser
import os


class JarvisConfig(configparser.ConfigParser):
    """
    Typed config manager that inherits from configparser.ConfigParser
    to maintain full backward compatibility while exposing typed accessors,
    standardizing env-var overrides, and providing unified fallback lookups.
    """

    def get_str(self, section: str, key: str, fallback: str = "") -> str:
        """Get config string, checking env-var overrides first."""
        env_key = f"JARVIS_{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        env_key_short = f"JARVIS_{key.upper()}"
        if env_key_short in os.environ:
            return os.environ[env_key_short]
        if not self.has_section(section):
            return fallback
        return self.get(section, key, fallback=fallback)

    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get config boolean, checking env-var overrides first."""
        env_key = f"JARVIS_{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            val = os.environ[env_key].lower()
            return val in ("true", "1", "yes", "on", "enable")
        env_key_short = f"JARVIS_{key.upper()}"
        if env_key_short in os.environ:
            val = os.environ[env_key_short].lower()
            return val in ("true", "1", "yes", "on", "enable")
        if not self.has_section(section):
            return fallback
        return self.getboolean(section, key, fallback=fallback)

    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """Get config integer, checking env-var overrides first."""
        env_key = f"JARVIS_{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            try:
                return int(os.environ[env_key])
            except ValueError:
                pass
        env_key_short = f"JARVIS_{key.upper()}"
        if env_key_short in os.environ:
            try:
                return int(os.environ[env_key_short])
            except ValueError:
                pass
        if not self.has_section(section):
            return fallback
        return self.getint(section, key, fallback=fallback)


def load_config(config_path: str) -> JarvisConfig:
    """
    Load INI config from an absolute path or relative to PROJECT_ROOT
    into JarvisConfig, with env-var resolution.
    """
    from core.runtime.paths import _resolve_path
    import logging
    
    log = logging.getLogger("jarvis.config")
    config = JarvisConfig()
    path = _resolve_path(config_path)

    if not path.exists():
        env = os.environ.get("JARVIS_ENV", "development").lower()
        msg = f"Config not found: {path}"
        if env == "production":
            log.critical(msg)
            raise SystemExit(2)  # CONFIG_ERROR
        log.warning("%s - using defaults", msg)
        return config

    try:
        with path.open("r", encoding="utf-8") as handle:
            config.read_file(handle)
    except configparser.Error as exc:
        log.critical("Config parse error: %s", exc)
        raise SystemExit(2) from exc
    except OSError as exc:
        log.critical("Config read error: %s", exc)
        raise SystemExit(2) from exc

    return config
