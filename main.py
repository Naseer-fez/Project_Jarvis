"""
╔══════════════════════════════════════════════════════════════╗
║               JARVIS V1 — TRUSTED CORE                       ║
║  Offline | Local | Deterministic | Human-in-the-loop         ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import logging
from core.controller import JarvisControllerV5

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("jarvis.main")


def main():
    logger.info("=" * 60)
    logger.info("JARVIS V1 STARTING — TRUSTED CORE ONLY")
    logger.info("=" * 60)

    controller = JarvisControllerV5()
    status = controller.initialize()

    logger.info(f"Session: {status['session_id']}")
    logger.info(f"Memory mode: {status['memory_mode']}")
    logger.info("Type 'help' for commands, 'exit' to quit.\n")

    # ── Main interaction loop ──────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            response = controller.process(user_input)

            if response == "__EXIT__":
                print("Jarvis: Goodbye.")
                break

            print(f"Jarvis: {response}\n")

        except KeyboardInterrupt:
            print("\n[Jarvis] Shutdown requested.")
            controller.shutdown()
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()