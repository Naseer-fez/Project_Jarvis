import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import subprocess
import threading

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

    print("[TEST] Server is up. Waiting 15 seconds for subsystems (LLM, embeddings) to load...", flush=True)
    time.sleep(15) 
    
    try:
        send_command("Open Notepad application.")
        time.sleep(10) # wait for notepad to open
        
        send_command("Type the exact text 'Hello world from Jarvis' into the active window (which should be Notepad).")
        time.sleep(5)
        
        send_command("Search the web for 'current population of Tokyo' and tell me the exact result.")
        time.sleep(5)
        
    finally:
        print("\n[TEST] Cleaning up, killing server...", flush=True)
        server.kill()
        print("[TEST] Done.", flush=True)

if __name__ == "__main__":
    run()
