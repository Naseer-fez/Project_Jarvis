import asyncio
import sys
import logging
from core.controller_v2 import JarvisControllerV2
from core.runtime.bootstrap import load_config, parse_args, apply_cli_overrides, DEFAULT_CONFIG_PATH

logging.basicConfig(level=logging.INFO)

async def run_test():
    print("Initializing Controller V2...")
    try:
        args = parse_args(sys.argv[1:])
        config = load_config(args.config if hasattr(args, "config") else DEFAULT_CONFIG_PATH)
        apply_cli_overrides(config, args)
        
        controller = JarvisControllerV2(config=config)
        await controller.initialize()

        print("\n--- Test: File Writing and Reading ---")
        prompt = "Please write 'Hello World' to a file named test.txt. Then read the file and tell me its contents."
        print(f"Sending prompt: {prompt}")
        
        response = await controller.process(prompt)
        print(f"\nJarvis Response:\n{response}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())
