"""
Jarvis - Agentic AI Assistant
Entry point: starts the agent loop with optional voice or text input.
"""

import asyncio
import argparse
import logging
import sys
from core.controller import MainController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/jarvis.log", mode="a"),
    ],
)

logger = logging.getLogger("Jarvis.Main")


async def main():
    parser = argparse.ArgumentParser(description="Jarvis Agentic Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice input/output")
    parser.add_argument("--autonomy", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Autonomy level: 0=chat, 1=suggest, 2=read-only, 3=write+confirm")
    parser.add_argument("--model", type=str, default="mistral",
                        help="Ollama model name (e.g. mistral, llama3, phi3)")
    args = parser.parse_args()

    controller = MainController(
        voice_enabled=args.voice,
        autonomy_level=args.autonomy,
        model=args.model,
    )

    logger.info("Starting Jarvis... (Ctrl+C to exit)")
    await controller.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Jarvis] Session ended.")

