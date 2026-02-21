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
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

# Use TYPE_CHECKING so we get IDE support without triggering circular imports at runtime
if TYPE_CHECKING:
    from core.controller import Controller

# Define the absolute root of the project based on this file's location
PROJECT_ROOT = Path(__file__).resolve().parent

def _load_config(config_path: str = "config/jarvis.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    path = PROJECT_ROOT / config_path
    
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
        help="Path to config file relative to project root (default: config/jarvis.ini)",
    )
    parser.add_argument(
        "--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()

async def _main(args: argparse.Namespace) -> None:
    config = _load_config(args.config)

    # Override log level from CLI if provided
    if args.log_level:
        if not config.has_section("logging"):
            config.add_section("logging")
        config.set("logging", "level", args.log_level.upper())

    # Initialise logging first
    import core.logger as logger_mod
    logger_mod.setup(config)
    log = logger_mod.get()

    # ── Audit verify mode ─────────────────────────────────────────────────────
    if args.verify:
        ok, count, err = logger_mod.verify_audit()
        if ok:
            log.info(f"✅ Audit log OK — {count} entries verified, chain intact.")
            print(f"✅ Audit log OK — {count} entries verified, chain intact.")
        else:
            log.error(f"❌ Audit log TAMPERED — {err}")
            print(f"❌ Audit log TAMPERED — {err}")
        sys.exit(0 if ok else 1)

    # ── Normal startup ────────────────────────────────────────────────────────
    from core.controller import Controller

    # Voice flag: --voice CLI arg OR jarvis.ini [voice] enabled=true
    voice_enabled = args.voice or config.getboolean("voice", "enabled", fallback=False)

    controller: Controller = Controller(config, voice=voice_enabled)

    try:
        await controller.start()
        await controller.run_cli()
    except asyncio.CancelledError:
        log.info("Async loop cancelled — shutting down")
    except Exception as e:
        log.critical(f"Fatal error in main loop: {e}\n{traceback.format_exc()}")
        print(f"❌ Fatal error: {e}")
    finally:
        await controller.shutdown()

def main() -> None:
    # Safely change to script directory so relative paths work globally
    os.chdir(PROJECT_ROOT)

    args = _parse_args()

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()
