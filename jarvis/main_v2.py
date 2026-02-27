"""
JARVIS v2 - Session 5: Voice & Automation Loop
Architecture: Trusted Core | Offline | Deterministic
Author: Jarvis Project
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from core.state_machine import JarvisStateMachine, State
from core.controller_v2 import JarvisController
from memory.hybrid_memory import HybridMemory
from voice.voice_layer import VoiceLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/jarvis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("JARVIS.Main")

BANNER = """
╔══════════════════════════════════════════════════════╗
║          J.A.R.V.I.S  -  Session 5 Online           ║
║   Just A Rather Very Intelligent System              ║
║   Mode: VOICE-FIRST | LOCAL | OFFLINE | TRUSTED      ║
╚══════════════════════════════════════════════════════╝
"""


async def main():
    print(BANNER)
    logger.info("Initializing Jarvis...")

    # Initialize core systems
    memory = HybridMemory(db_path="jarvis_memory.db")
    await memory.initialize()

    state_machine = JarvisStateMachine()
    controller = JarvisController(state_machine=state_machine, memory=memory)
    await controller.initialize()

    voice = VoiceLayer(controller=controller, memory=memory)

    print("\nChoose interface mode:")
    print("  [1] Voice Mode (Wake Word: 'Jarvis')")
    print("  [2] CLI Mode (Text Input)")
    print("  [3] Hybrid Mode (Both)")

    mode = input("\nSelect mode [1/2/3]: ").strip()

    if mode == "1":
        logger.info("Starting Voice Mode...")
        await voice.run_voice_loop()
    elif mode == "2":
        logger.info("Starting CLI Mode...")
        await controller.run_cli_loop()
    else:
        logger.info("Starting Hybrid Mode...")
        await asyncio.gather(
            voice.run_voice_loop(),
            controller.run_cli_loop()
        )


if __name__ == "__main__":
    asyncio.run(main())
