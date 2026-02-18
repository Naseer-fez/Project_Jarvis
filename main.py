"""
JARVIS SESSION 5 ENTRY POINT
Updates: Uses ControllerV3, handles Memory Intelligence output.
"""
import sys
import logging
from core.controller import JarvisControllerV3

# Configure logging to file only, keep console clean for UI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("jarvis_session5.log")]
)

def main():
    print("╔════════════════════════════════════════════╗")
    print("║     JARVIS SESSION 5 - INTELLIGENT         ║")
    print("║   (Memory Gate & Self-Reflection Active)   ║")
    print("╚════════════════════════════════════════════╝")
    
    try:
        jarvis = JarvisControllerV3()
        status = jarvis.initialize()
    except Exception as e:
        print(f"\nCRITICAL INIT FAILURE: {e}")
        return

    print(f"• Session ID: {status.get('session_id')}")
    print(f"• Memory Mode: {status.get('memory_mode')}")
    print(f"• System: Online\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            
            # Special handling for exit to ensure shutdown hook runs
            if user_input.lower() in ("exit", "quit"):
                jarvis.shutdown()
                print("Goodbye.")
                break

            print("Jarvis: ", end="", flush=True)
            
            response_generator = jarvis.process(user_input, stream=True)
            
            # Handle both string (commands) and generators (LLM)
            if isinstance(response_generator, str):
                print(response_generator)
            else:
                full_resp = ""
                for chunk in response_generator:
                    print(chunk, end="", flush=True)
                    full_resp += chunk
                print() # Newline after stream
                
        except KeyboardInterrupt:
            jarvis.shutdown()
            print("\nForce Exit.")
            break
        except Exception as e:
            logging.error(f"Runtime error: {e}")
            print(f"\n[System Error]: {e}")

if __name__ == "__main__":
    main()