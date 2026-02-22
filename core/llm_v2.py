"""
core/llm_v2.py
───────────────────
LLM Client for V2 (Session 4) with memory injection.
"""
import requests

class LLMClientV2:
    def __init__(self, model_name="deepseek-r1:8b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def chat(self, prompt: str, system_context: str = "") -> str:
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": f"You are Jarvis. Use this memory:\n{system_context}"},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "")
            
            # Treat 500s or any other status as Offline to trigger fallback logic
            return "LLM Offline."
        except requests.exceptions.RequestException:
            return "LLM Offline."