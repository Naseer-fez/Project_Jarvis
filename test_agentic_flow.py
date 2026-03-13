import asyncio
from core.controller_v2 import JarvisControllerV2
import logging
logging.basicConfig(level=logging.INFO)

async def main():
    print("Initializing Controller V2...")
    controller = JarvisControllerV2()
    
    # Required for initializing the dashboard logic and other properties
    controller.initialize()

    text = "Look up the current weather in New York and fetch the top news headline there today."
    print(f"Sending input: {text}")
    response = await controller.process(text)
    print(f"\nJarvis Response:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
