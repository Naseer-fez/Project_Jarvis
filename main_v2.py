"""
main_v2.py
───────────────────
Terminal interface for Jarvis Session 4 (Semantic Memory).
"""
import sys
from core.controller_v2 import JarvisControllerV2

def main():
    print("="*50)
    print("🤖 JARVIS V2 - SEMANTIC MEMORY MODE")
    print("="*50)
    
    ctrl = JarvisControllerV2()
    status = ctrl.initialize()
    
    print(f"✅ Initialized Session: {status.get('session_id')}")
    print(f"🧠 Memory Mode: {status.get('memory_mode').upper()}")
    print("Type 'help' for commands, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = ctrl.process(user_input)
            print(f"Jarvis: {response}\n")
            
        except KeyboardInterrupt:
            break

    print("\nSaving session and shutting down...")
    summary = ctrl.session_summary()
    print(f"Session closed. Exchanges: {summary['exchanges']}")

if __name__ == "__main__":
    main()