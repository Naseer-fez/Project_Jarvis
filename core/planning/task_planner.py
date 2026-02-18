"""
core/planning/task_planner.py
══════════════════════════════
LLM-powered planner using DeepSeek R1:8b via Ollama.

V1 Rules:
  - Input: user intent (text)
  - Output: validated JSON plan
  - NO execution. NO side effects. Planning ONLY.
  - If LLM output is invalid JSON → return error plan, never crash
  - Plans must pass RiskEvaluator before any step is executed
  - All plans logged
"""

import json
import asyncio
import aiohttp
from datetime import datetime, timezone
from core.logger import get_logger, audit
from core.risk_evaluator import RiskEvaluator

logger = get_logger("planner")

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-r1:8b"

SYSTEM_PROMPT = """You are the planning module of Jarvis, a local AI assistant.

Your ONLY job is to convert a user's intent into a structured JSON execution plan.

STRICT RULES:
1. Output ONLY valid JSON — no markdown, no explanation, no preamble
2. Never include physical actions (mouse, keyboard, robotics) — these are BLOCKED in V1
3. Available tools in V1:
   - vision.analyze_image
   - vision.snapshot
   - memory.read
   - memory.write
   - planner.generate_plan (meta-planning only)
4. If the request cannot be completed with available tools, set "feasible": false
5. Every step needs: id, action, description, requires_confirmation (bool)

OUTPUT FORMAT (strict):
{
  "goal": "string — what the user wants",
  "feasible": true,
  "reasoning": "string — brief reasoning",
  "steps": [
    {
      "id": 1,
      "action": "tool.method",
      "description": "what this step does",
      "requires_confirmation": false,
      "params": {}
    }
  ]
}

If not feasible:
{
  "goal": "string",
  "feasible": false,
  "reasoning": "string — why it cannot be done",
  "steps": []
}
"""


def _build_user_prompt(intent: str, context: str = "") -> str:
    ctx_block = f"\n\nCONTEXT:\n{context}" if context else ""
    return f"USER INTENT: {intent}{ctx_block}\n\nGenerate the JSON plan now."


class TaskPlanner:
    """
    Calls DeepSeek R1:8b to generate a structured execution plan.
    Plans are validated and risk-checked before being returned.
    """

    def __init__(self):
        self.risk = RiskEvaluator()

    async def plan(self, intent: str, context: str = "") -> dict:
        """
        Generate a plan for the given intent.
        Returns validated plan dict. Never raises — returns error plan on failure.
        """
        logger.info(f"PLANNING: intent={intent!r}")

        raw_response = await self._call_llm(intent, context)
        plan = self._parse_plan(raw_response, intent)
        plan = self._risk_check_plan(plan)

        audit(
            logger,
            f"PLAN_GENERATED: goal={plan.get('goal')} feasible={plan.get('feasible')} steps={len(plan.get('steps', []))}",
            plan=plan.get("goal"),
            action="plan_generated"
        )
        return plan

    async def _call_llm(self, intent: str, context: str) -> str:
        """Call Ollama DeepSeek R1:8b. Returns raw string response."""
        payload = {
            "model": LLM_MODEL,
            "prompt": _build_user_prompt(intent, context),
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for deterministic planning
                "top_p": 0.9,
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        logger.error(f"LLM call failed: HTTP {resp.status}")
                        return ""
                    data = await resp.json()
                    return data.get("response", "")
        except aiohttp.ClientConnectorError:
            logger.error("Ollama not reachable. Is it running? (ollama serve)")
            return ""
        except asyncio.TimeoutError:
            logger.error("LLM call timed out after 60s")
            return ""
        except Exception as e:
            logger.error(f"LLM call exception: {e}")
            return ""

    def _parse_plan(self, raw: str, original_intent: str) -> dict:
        """Parse LLM response into validated plan dict."""
        if not raw.strip():
            return self._error_plan(original_intent, "LLM returned empty response")

        # Strip <think>...</think> blocks from DeepSeek R1
        import re
        raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code fences if present
        if raw_clean.startswith("```"):
            raw_clean = re.sub(r"^```[a-z]*\n?", "", raw_clean)
            raw_clean = re.sub(r"\n?```$", "", raw_clean)

        try:
            plan = json.loads(raw_clean)
        except json.JSONDecodeError as e:
            logger.error(f"LLM output is not valid JSON: {e}\nRaw output: {raw[:300]}")
            return self._error_plan(original_intent, f"Invalid JSON from LLM: {e}")

        # Validate required fields
        required = {"goal", "feasible", "steps"}
        missing = required - set(plan.keys())
        if missing:
            logger.error(f"Plan missing required fields: {missing}")
            return self._error_plan(original_intent, f"Plan missing fields: {missing}")

        # Validate each step
        for step in plan.get("steps", []):
            for field in ("id", "action", "description"):
                if field not in step:
                    logger.warning(f"Step {step} missing field '{field}' — plan may be incomplete")

        logger.info(f"Plan parsed successfully: goal={plan.get('goal')!r} steps={len(plan.get('steps', []))}")
        return plan

    def _risk_check_plan(self, plan: dict) -> dict:
        """Run RiskEvaluator on every step. Block unsafe plans."""
        steps = plan.get("steps", [])
        if not steps:
            return plan

        all_safe, results = self.risk.evaluate_plan(steps)

        if not all_safe:
            blocked_tools = [r.tool for r in results if not r.allowed]
            logger.warning(f"Plan contains blocked tools: {blocked_tools}. Marking infeasible.")
            plan["feasible"] = False
            plan["risk_blocked"] = True
            plan["blocked_tools"] = blocked_tools
            plan["steps"] = []
            plan["reasoning"] = (
                f"Plan was blocked by RiskEvaluator. "
                f"Blocked tools: {blocked_tools}. "
                f"These tools are not permitted in V1."
            )

        return plan

    def _error_plan(self, intent: str, reason: str) -> dict:
        """Return a safe fallback error plan."""
        return {
            "goal": intent,
            "feasible": False,
            "reasoning": f"Planning failed: {reason}. Jarvis cannot proceed.",
            "steps": [],
            "error": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
