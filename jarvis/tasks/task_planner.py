"""
JARVIS Task Planner - Session 5
Uses DeepSeek-R1:8b via local Ollama to generate deterministic JSON plans.
NO cloud APIs. All inference is local.
"""

import json
import logging
import re
import httpx
from typing import Optional

logger = logging.getLogger("JARVIS.TaskPlanner")

OLLAMA_URL = "http://localhost:11434/api/generate"
PLANNING_MODEL = "deepseek-r1:8b"

SYSTEM_PROMPT = """You are the planning core of JARVIS, an offline AI assistant.
Your ONLY job is to convert user intent into a structured JSON task plan.

RULES:
1. Output ONLY valid JSON — no markdown, no explanations, no preamble.
2. Never suggest cloud APIs, internet calls, or external services.
3. All actions must be local and deterministic.
4. If the request is unclear, output a clarification plan.

JSON Schema:
{
  "intent": "brief description of what user wants",
  "confidence": 0.0-1.0,
  "steps": [
    {
      "step_id": 1,
      "action": "action_name",
      "description": "what this step does",
      "parameters": {},
      "risk_level": "LOW|MEDIUM|HIGH"
    }
  ],
  "requires_clarification": false,
  "clarification_question": null
}

Available actions: read_file, write_file, search_memory, open_app, 
list_directory, run_script, get_system_info, search_web_local, 
play_music, set_reminder, clarify_intent
"""


class TaskPlanner:
    def __init__(self, model: str = PLANNING_MODEL, timeout: int = 60):
        self.model = model
        self.timeout = timeout
        logger.info(f"TaskPlanner initialized with model: {model}")

    async def plan(self, user_intent: str, context: str = "") -> dict:
        """Generate a JSON task plan from natural language intent."""
        prompt = f"User request: {user_intent}"
        if context:
            prompt += f"\nContext: {context}"

        logger.info(f"Planning for intent: '{user_intent}'")

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for deterministic planning
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(OLLAMA_URL, json=payload)
                response.raise_for_status()
                data = response.json()

            raw_text = data.get("response", "")
            plan = self._parse_json(raw_text)
            logger.info(f"Plan generated: {len(plan.get('steps', []))} steps, confidence={plan.get('confidence', 0)}")
            return plan

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Is it running? Run: ollama serve")
            return self._error_plan("Ollama service unavailable. Run 'ollama serve' first.")
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._error_plan(str(e))

    def _parse_json(self, text: str) -> dict:
        """Extract and parse JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from response: {text[:200]}")
        return self._error_plan("LLM returned invalid JSON format.")

    def _error_plan(self, reason: str) -> dict:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "steps": [],
            "requires_clarification": True,
            "clarification_question": f"I encountered an issue: {reason}",
            "error": reason
        }
