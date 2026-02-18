"""
JARVIS ENTRY POINT
Session 4 compatible
"""
import sys
import logging
from core.controller import JarvisControllerV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("jarvis.log"), logging.StreamHandler(sys.stdout)]
)
# Silence loud libraries
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def main():
    print("╔══════════════════════════════════════╗")
    print("║      JARVIS SESSION 4 (HYBRID)       ║")
    print("╚══════════════════════════════════════╝")
    
    # 1. Initialize Controller
    print("• Initializing core systems...")
    try:
        jarvis = JarvisControllerV2()
        status = jarvis.initialize()
    except Exception as e:
        print(f"\nCRITICAL INIT FAILURE: {e}")
        print("Check your imports and file structure.")
        return

    print(f"• Session ID: {status.get('session_id')}")
    print(f"• Memory Mode: {status.get('memory_mode')}")
    print(f"• LLM Status: {'ONLINE' if status.get('ollama') else 'OFFLINE (Check Ollama)'}")
    print("\nReady. Type 'exit' to quit.\n")

    # 2. Main Loop
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            # Use streaming for better UX
            response = jarvis.process(user_input, stream=True)
            
            if response == "__EXIT__":
                print("Jarvis: Goodbye!")
                break
                
            if not response:
                print("Jarvis: ...")
                
        except KeyboardInterrupt:
            print("\nForce Exit.")
            break
        except Exception as e:
            logging.error(f"Runtime error: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()