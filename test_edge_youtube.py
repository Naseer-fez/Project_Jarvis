import time
import sys
import requests
import subprocess
import threading

sys.stdout.reconfigure(encoding='utf-8')

TOKEN = "jarvis"
URL = "http://localhost:7070/command"
HEADERS = {"X-Dashboard-Token": TOKEN, "Content-Type": "application/json"}

def send_command(text):
    print(f"\n[TEST] Sending: {text}", flush=True)
    try:
        resp = requests.post(URL, headers=HEADERS, json={"text": text}, timeout=300)
        if resp.status_code == 200:
            print(f"[TEST] Response: {resp.json().get('response', '')}", flush=True)
        else:
            print(f"[TEST] Error: {resp.status_code} - {resp.text}", flush=True)
    except Exception as e:
        print(f"[TEST] Exception: {e}", flush=True)

def run():
    print("[TEST] Starting Jarvis Server...", flush=True)
    server = subprocess.Popen(
        [r"jarvis_env\Scripts\python.exe", "main.py", "--headless", "--gui"],
    )
    
    # Wait for server
    print("[TEST] Waiting for server health check...", flush=True)
    up = False
    for i in range(120):
        try:
            if requests.get("http://localhost:7070/health", timeout=2).status_code == 200:
                up = True
                break
        except:
            time.sleep(1)
            
    if not up:
        print("[TEST] Server failed to start.", flush=True)
        server.kill()
        return

    print("[TEST] Server is up. Waiting 15 seconds for subsystems...", flush=True)
    time.sleep(15) 
    
    try:
        # Command 1: Open Edge
        send_command("Launch the Microsoft Edge application.")
        time.sleep(5)
        
        # Command 2: Go to YouTube and search
        # Giving it explicit step-by-step instructions so the LLM knows how to use its tools for this specific GUI flow.
        send_command("In the currently open Microsoft Edge window, type 'youtube.com' and press Enter to navigate to YouTube. Wait a few seconds, then press 'tab' multiple times or use '/' to focus the search bar, type 'cute cats', and press Enter to search.")
        time.sleep(5)
        
    finally:
        print("\n[TEST] Cleaning up, killing server...", flush=True)
        server.kill()
        print("[TEST] Done.", flush=True)

if __name__ == "__main__":
    run()
