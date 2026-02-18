"""
TaskPlanner — decomposes a user goal into an ordered list of steps.
Uses Ollama (local LLM) to generate structured JSON plans.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
import httpx

logger = logging.getLogger("Jarvis.TaskPlanner")

PLAN_SYSTEM_PROMPT = """You are a planning module for an AI assistant called Jarvis.
Given a user goal, decompose it into a minimal, ordered list of concrete steps.

Respond ONLY with valid JSON in this exact schema — no explanation, no markdown:
{
  "goal": "<original goal>",
  "steps": [
    {
      "step_id": 1,
      "description": "<what to do>",
      "tool_required": "<tool_name or null>",
      "arguments": {"key": "value"},
      "expected_observation": "<what success looks like>",
      "reversible": true,
      "estimated_risk": 0.1
    }
  ]
}

Available tools: get_time, get_system_stats, list_directory, read_file, write_file_safe, search_memory, log_event
If no tool is needed for a step, set tool_required to null and arguments to null.
Limit steps to 10 maximum. Keep it concise and practical.
"""


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_required: Optional[str]
    arguments: Optional[dict]
    expected_observation: str
    reversible: bool
    estimated_risk: float


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    raw_json: Optional[dict] = field(default=None, repr=False)

    def __len__(self):
        return len(self.steps)

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}", f"Steps ({len(self.steps)}):"]
        for s in self.steps:
            tool = f" [tool: {s.tool_required}]" if s.tool_required else ""
            lines.append(f"  {s.step_id}. {s.description}{tool}")
        return "\n".join(lines)


class TaskPlanner:
    def __init__(self, model: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url

    async def plan(self, goal: str, context: str = "") -> Optional[Plan]:
        """
        Ask the local LLM to generate a structured plan for the goal.
        Returns a Plan object, or None on failure.
        """
        user_prompt = f"User goal: {goal}"
        if context:
            user_prompt += f"\n\nContext:\n{context}"

        logger.info(f"Planning goal: {goal!r}")

        raw = await self._call_ollama(PLAN_SYSTEM_PROMPT, user_prompt)
        if not raw:
            logger.error("TaskPlanner: No response from Ollama.")
            return None

        return self._parse_plan(raw)

    async def _call_ollama(self, system: str, user: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Is it running? (ollama serve)")
            return None
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None

    def _parse_plan(self, raw_text: str) -> Optional[Plan]:
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", raw_text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object from surrounding text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error("Failed to parse plan JSON from LLM response.")
                    logger.debug(f"Raw text: {raw_text[:500]}")
                    return None
            else:
                logger.error("No JSON object found in LLM response.")
                return None

        try:
            steps = [
                PlanStep(
                    step_id=s["step_id"],
                    description=s["description"],
                    tool_required=s.get("tool_required"),
                    arguments=s.get("arguments"),
                    expected_observation=s.get("expected_observation", ""),
                    reversible=s.get("reversible", True),
                    estimated_risk=float(s.get("estimated_risk", 0.1)),
                )
                for s in data.get("steps", [])
            ]

            plan = Plan(goal=data.get("goal", ""), steps=steps, raw_json=data)
            logger.info(f"Plan created: {len(steps)} step(s)")
            logger.debug(plan.summary())
            return plan

        except (KeyError, TypeError) as e:
            logger.error(f"Plan schema error: {e}")
            return None

