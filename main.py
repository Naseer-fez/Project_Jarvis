"""
main.py — Jarvis V2 entry point.

Usage:
  python main.py               # CLI only (V1 mode)
  python main.py --voice       # Voice + CLI hybrid (V2 mode)
  python main.py --verify      # Verify audit log integrity and exit
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import os
import sys
from pathlib import Path


def _load_config(config_path: str = "config/jarvis.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    path = Path(config_path)
    if path.exists():
        config.read(path, encoding="utf-8")
    else:
        print(f"⚠  Config not found at {path} — using defaults")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jarvis V2 — Offline Local AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Start in CLI-only mode (V1 behaviour)
  python main.py --voice          Start with voice loop enabled (V2)
  python main.py --verify         Verify audit log integrity
  python main.py --config my.ini  Use custom config file
        """,
    )
    parser.add_argument(
        "--voice", action="store_true",
        help="Enable V2 voice loop (wake word + STT + TTS)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify audit log integrity and exit",
    )
    parser.add_argument(
        "--config", default="config/jarvis.ini",
        help="Path to config file (default: config/jarvis.ini)",
    )
    parser.add_argument(
        "--log-level", default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    config = _load_config(args.config)

    # Override log level from CLI if provided
    if args.log_level:
        config.set("logging", "level", args.log_level.upper())

    # Initialise logging first
    import core.logger as logger_mod
    logger_mod.setup(config)
    log = logger_mod.get()

    # ── Audit verify mode ─────────────────────────────────────────────────────
    if args.verify:
        ok, count, err = logger_mod.verify_audit()
        if ok:
            print(f"✅ Audit log OK — {count} entries verified, chain intact.")
        else:
            print(f"❌ Audit log TAMPERED — {err}")
        sys.exit(0 if ok else 1)

    # ── Normal startup ────────────────────────────────────────────────────────
    from core.controller import Controller

    # Voice flag: --voice CLI arg OR jarvis.ini [voice] enabled=true
    voice_enabled = args.voice or config.getboolean("voice", "enabled", fallback=False)

    controller = Controller(config, voice=voice_enabled)

    try:
        await controller.start()
        await controller.run_cli()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
    finally:
        await controller.shutdown()


def main() -> None:
    # Change to script directory so relative paths work
    os.chdir(Path(__file__).parent)

    args = _parse_args()

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
