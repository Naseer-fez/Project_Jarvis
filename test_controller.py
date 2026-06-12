import asyncio
import sys
import logging
from core.runtime.bootstrap import load_config
from core.controller_v2 import JarvisControllerV2

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

async def main():
    print("Loading config...", flush=True)
    config = load_config("config/jarvis.ini")
    print("Creating controller...", flush=True)
    controller = JarvisControllerV2(config=config)
    print("Starting controller...", flush=True)
    await controller.startup()
    print("Sending input...", flush=True)
    response = await controller.process("What time is it?")
    print(f"Response: {response}", flush=True)
    await controller.shutdown()
    print("Done.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
