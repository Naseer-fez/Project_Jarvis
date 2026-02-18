"""
╔══════════════════════════════════════════════════════════════╗
║               JARVIS V1 — TRUSTED CORE                       ║
║  Offline | Local | Deterministic | Human-in-the-loop         ║
╚══════════════════════════════════════════════════════════════╝

Entry point. Starts the async core controller.
V1 scope: Perception, Memory, Planning, Vision only.
NO voice. NO desktop automation. NO robotics.
"""

import asyncio
import sys
from core.controller import JarvisController
from core.logger import get_logger

logger = get_logger("main")


async def main():
    logger.info("=" * 60)
    logger.info("JARVIS V1 STARTING — TRUSTED CORE ONLY")
    logger.info("Authority ceiling: L1_INTERPRET (no physical actions)")
    logger.info("=" * 60)

    controller = JarvisController()
    await controller.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[JARVIS] Shutdown requested by user. Goodbye.")
        sys.exit(0)
