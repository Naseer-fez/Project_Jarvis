"""Asynchronous task planner to generate execution plans."""

from __future__ import annotations

import json
import logging
import re
import inspect
from typing import Any

from core.autonomy.risk_evaluator import RiskLevel, RiskEvaluator

# SYSTEM_TOOL_SCHEMA removed in favor of CapabilityRegistry dynamic discovery

_GUI_TOOL_NAMES = {
    "click",
    "double_click",
    "right_click",
    "click_text_on_screen",
    "click_screen_target",
    "double_click_screen_target",
    "right_click_screen_target",
    "move_mouse",
    "scroll",
    "drag",
    "type_text",
    "press_key",
    "hotkey",
    "focus_window",
    "clipboard_get",
    "clipboard_set",
    "clipboard_paste",
}


def _strip_planner_artifacts(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()

logger = logging.getLogger("Jarvis.Planner")


class TaskPlanner:
    def __init__(self, config=None, llm=None, registry=None) -> None:
        self.config = config
        self.risk_evaluator = RiskEvaluator(config)
        self.llm = llm
        self.registry = registry

    def _tool_schema(self) -> dict[str, list[dict[str, Any]]]:
        tools = []
        if self.registry:
            for name in self.registry.registered_tools():
                cap = self.registry.get(name)
                desc = getattr(cap, "description", "") or f"Execute {name}"
                schema: dict[str, Any] = {"name": name, "description": desc}
                
                if hasattr(cap, "handler") and cap.handler:
                    try:
                        sig = inspect.signature(cap.handler)
                        params = {}
                        for param_name, param in sig.parameters.items():
                            if param_name == "context":
                                continue
                            param_info: dict[str, Any] = {"type": "string"}
                            if param.annotation != inspect.Parameter.empty:
                                if hasattr(param.annotation, "__name__"):
                                    param_info["type"] = param.annotation.__name__
                                else:
                                    param_info["type"] = str(param.annotation)
                            if param.default != inspect.Parameter.empty:
                                if isinstance(param.default, (str, int, float, bool, type(None))):
                                    param_info["default"] = param.default
                                else:
                                    param_info["default"] = str(param.default)
                            else:
                                param_info["required"] = True
                            params[param_name] = param_info
                        schema["parameters"] = params
                    except Exception:
                        pass
                tools.append(schema)

        allow_gui = False
        try:
            if self.config:
                allow_gui = self.config.getboolean(
                    "execution",
                    "allow_gui_automation",
                    fallback=False,
                )
        except Exception:
            allow_gui = False

        if not allow_gui:
            tools = [tool for tool in tools if tool["name"] not in _GUI_TOOL_NAMES]

        return {"tools": tools}

    async def _call_ollama(self, prompt: str) -> str:
        if not self.llm or not hasattr(self.llm, "complete"):
            return ""
        try:
            res = self.llm.complete(prompt, task_type="tool_picker")
            if inspect.isawaitable(res):
                return str(await res)
            return str(res)
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            return ""

    async def plan(self, user_input: str, context: str = "") -> dict[str, Any]:
        text = str(user_input or "").strip()
        logger.info("Starting task planning", extra={"metadata": {"intent_length": len(text), "context_length": len(context)}})
        raw = await self._call_ollama(self._build_prompt(text, context))

        if raw:
            logger.info("Raw LLM output: %s", raw)
            parsed = self._parse_llm_plan(raw)
            if parsed is not None:
                logger.info("Successfully parsed plan from LLM output", extra={"metadata": {"confidence": parsed.get("confidence")}})
                return self._enrich_plan(text, parsed)
            logger.warning("Failed to parse LLM plan, falling back to clarification")
            return self._clarification_plan(text)

        logger.warning("Empty LLM response, falling back to basic plan")
        return self._enrich_plan(text, self._fallback_plan(text))

    def _build_prompt(self, user_input: str, context: str) -> str:
        schema_format = {
            "intent": "user request",
            "summary": "overall plan summary",
            "confidence": 0.9,
            "steps": [
                {
                    "id": 1,
                    "action": "tool_name",
                    "description": "why we call this tool",
                    "parameters": {"<argument_name>": "<argument_value>"}
                }
            ],
            "clarification_needed": False,
            "clarification_prompt": ""
        }
        
        example_json = {
            "intent": "read a file",
            "summary": "I will use read_file",
            "confidence": 0.99,
            "steps": [
                {
                    "id": 1,
                    "action": "read_file",
                    "description": "Read the config file",
                    "parameters": {"path": "/etc/config.json"}
                }
            ],
            "clarification_needed": False,
            "clarification_prompt": ""
        }

        return (
            "You are a task planner. Create a step-by-step action plan using the available tools to satisfy the user request.\n"
            f"User request: {user_input}\n"
            f"Context: {context}\n"
            f"Available tools: {json.dumps(self._tool_schema())}\n\n"
            "You MUST return a valid JSON object matching the following structure exactly:\n"
            f"{json.dumps(schema_format, indent=2)}\n\n"
            "CRITICAL: For EVERY tool step in 'steps', you MUST include a 'parameters' dictionary containing the required arguments. "
            "The keys in 'parameters' MUST exactly match the argument names shown in the tool's schema.\n\n"
            f"Example Output:\n{json.dumps(example_json, indent=2)}\n\n"
            "Return ONLY the strict JSON object. No explanations, no markdown block, no extra text."
        )

    def _parse_llm_plan(self, raw: str) -> dict[str, Any] | None:
        cleaned = _strip_planner_artifacts(raw)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt basic repair for missing commas (a common issue with smaller models)
            import re
            cleaned = re.sub(r'"\s*\n\s*"', '",\n"', cleaned)
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _fallback_plan(self, text: str) -> dict[str, Any]:
        return self._clarification_plan(text)

    def _clarification_plan(self, text: str) -> dict[str, Any]:
        return {
            "intent": text,
            "summary": "Need clarification before taking action.",
            "confidence": 0.2,
            "steps": [],
            "clarification_needed": True,
            "clarification_prompt": "Please clarify what you want me to do.",
        }

    def _enrich_plan(self, text: str, plan: dict[str, Any]) -> dict[str, Any]:
        steps_list = self._normalize_steps(plan.get("steps", []))
        normalized: dict[str, Any] = {
            "intent": str(plan.get("intent", text)),
            "summary": str(plan.get("summary", "")).strip(),
            "confidence": float(plan.get("confidence", 0.0) or 0.0),
            "steps": steps_list,
            "clarification_needed": bool(plan.get("clarification_needed", False)),
            "clarification_prompt": str(plan.get("clarification_prompt", "") or ""),
        }

        tools_required = [
            step["action"] for step in steps_list if step.get("action")
        ]
        risk = self.risk_evaluator.evaluate(tools_required)
        risk_label = risk.level.label().lower()
        if risk.level >= RiskLevel.CRITICAL:
            risk_label = "critical"
        elif risk.level >= RiskLevel.HIGH:
            risk_label = "high"
        elif risk.level >= RiskLevel.MEDIUM:
            risk_label = "medium"
        else:
            risk_label = "low"

        normalized["tools_required"] = tools_required
        normalized["risk_level"] = risk_label
        normalized["confirmation_required"] = risk.requires_confirmation
        logger.debug("Plan enriched with risk evaluation", extra={"metadata": {"risk_level": risk_label, "tools_count": len(tools_required)}})
        return normalized

    def _normalize_steps(self, steps: Any) -> list[dict[str, Any]]:
        if not isinstance(steps, list):
            return []
        normalized: list[dict[str, Any]] = []
        step_list: list[Any] = steps
        for idx, step in enumerate(step_list, start=1):
            if not isinstance(step, dict):
                continue
            params = step.get("params", {})
            normalized.append(
                {
                    "id": int(step.get("id", idx)),
                    "action": str(step.get("action", "")).strip(),
                    "description": str(step.get("description", "")).strip(),
                    "params": params if isinstance(params, dict) else {},
                }
            )
        return normalized
