"""
JARVIS SESSION 6 ENTRY POINT
Updates: Uses ControllerV4 (Safety & Discipline).
"""
import logging
from core.controller import JarvisControllerV4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("jarvis_session6.log")]
)

def main():
    print("╔════════════════════════════════════════════╗")
    print("║     JARVIS SESSION 6 - DISCIPLINED         ║")
    print("║   (Safety Gate & Intent Router Active)     ║")
    print("╚════════════════════════════════════════════╝")
    
    jarvis = JarvisControllerV4()
    status = jarvis.initialize()

    print(f"• Session ID: {status.get('session_id')}")
    print(f"• Safety Gate: {status.get('safety_gate')}")
    print("• System: Online\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            
            # Controller handles exit commands internally now via routing
            
            print("Jarvis: ", end="", flush=True)
            
            response = jarvis.process(user_input, stream=True)
            
            if response == "__EXIT__":
                print("Shutting down...")
                break
            
            # Handle string response (Blockers/Commands) vs Generator (Chat)
            if isinstance(response, str):
                print(response)
            else:
                full_resp = ""
                for chunk in response:
                    print(chunk, end="", flush=True)
                    full_resp += chunk
                print() 
                
        except KeyboardInterrupt:
            jarvis.shutdown()
            print("\nForce Exit.")
            break
        except Exception as e:
            logging.error(f"Runtime error: {e}")
            print(f"\n[System Error]: {e}")

if __name__ == "__main__":
    main()