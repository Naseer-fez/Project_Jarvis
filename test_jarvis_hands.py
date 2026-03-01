import asyncio
import logging
import os

# Import the core components we just built
from core.agentic.autonomy_policy import AutonomyPolicy
from core.agentic.belief_state import BeliefState
from core.execution.dispatcher import Dispatcher

# Suppress debug noise to keep the output clean
logging.basicConfig(level=logging.WARNING)

# ── MOCK DEPENDENCIES ──────────────────────────────────────────

class MockReflection:
    """Fakes the reflection engine so we can see what Jarvis learns."""
    def record_action(self, payload):
        print(f"\n🧠 [Reflection Engine] Jarvis learned:")
        print(f"   Tool: {payload['tool']}")
        print(f"   Success: {payload['success']}")
        print(f"   Output: {payload['output']}")

class MockVoiceLayer:
    """Fakes the microphone so you can just type 'y' or 'n' in the console."""
    async def ask_confirm(self, prompt: str) -> bool:
        print(f"\n🔊 [TTS Speaks]: '{prompt}'")
        ans = input("🎤 [Mic Input] (type 'y' to approve, 'n' to reject): ").strip().lower()
        return ans == 'y'

# ── MAIN TEST LOOP ─────────────────────────────────────────────

async def run_demo():
    print("==================================================")
    print(" 🤖 BOOTING JARVIS V3 EXECUTION TEST")
    print("==================================================\n")

    # 1. Initialize the Brain's safety rules
    # beliefs = BeliefState() 
    policy = AutonomyPolicy()
    reflection = MockReflection()
    voice = MockVoiceLayer()

    # 2. Initialize the Hands
    dispatcher = Dispatcher(
        autonomy_policy=policy, 
        reflection_engine=reflection, 
        voice_layer=voice
    )

    # 3. Craft a fake action that DeepSeek WOULD generate
    test_file_path = "D:/AI/Jarvis/IT_WORKS.txt"
    
    action = {
        "tool": "write_file",
        "args": {
            "path": test_file_path,
            "content": "Hello Sir. My hands are fully operational. I have successfully written to the filesystem.",
            "overwrite": True
        },
        "rationale": "Demonstrating to the user that I can manipulate the filesystem."
    }

    print(f"🎯 INTENT: The Planner has decided to create a file.")
    print(f"📦 ACTION PAYLOAD: {action['tool']} -> {action['args']['path']}")
    print(f"💡 RATIONALE: {action['rationale']}\n")
    
    print("⏳ Dispatching to Execution Layer...")
    
    # 4. Execute!
    result = await dispatcher.dispatch(action)

    print("\n==================================================")
    print(" ✅ EXECUTION RESULT")
    print("==================================================")
    if result.success:
        print(f"🎉 SUCCESS! Check your folder for: {test_file_path}")
        print(f"📄 File contents: '{result.output}'")
    else:
        print(f"❌ FAILED or BLOCKED.")
        print(f"⚠️ Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(run_demo())