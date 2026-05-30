from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_GOAL_CREATE_KEYWORDS = (
    "remind me",
    "set goal",
    "schedule",
    "don't forget",
    "remember to",
)
_GOAL_STRIP_KEYWORDS = (
    "remind me to",
    "set goal",
    "schedule",
    "don't forget to",
    "remember to",
)
_GOAL_LIST_KEYWORDS = ("what are my goals", "show goals", "list goals", "my goals")


@dataclass(frozen=True)
class GoalIntentResult:
    response: str
    mutated: bool = False


_neural_scheduler = None


def get_neural_scheduler() -> Any:
    global _neural_scheduler
    if _neural_scheduler is None:
        try:
            from core.agentic.neural_scheduler import NeuralScheduler
            _neural_scheduler = NeuralScheduler()
        except Exception as exc:
            logger.warning("Failed to load neural scheduler: %s", exc)
    return _neural_scheduler


def handle_goal_intent(
    text: str,
    user_input: str,
    *,
    goal_manager: Any,
    scheduler: Any,
) -> GoalIntentResult | None:
    if any(keyword in text for keyword in _GOAL_CREATE_KEYWORDS):
        description = user_input
        for keyword in _GOAL_STRIP_KEYWORDS:
            description = re.sub(
                re.escape(keyword),
                "",
                description,
                flags=re.IGNORECASE,
            ).strip()
        description = description.strip(" .?!")
        if description:
            # Predict priority and delay using neural scheduler
            pred_priority, pred_delay = 5, 0.0
            ns = get_neural_scheduler()
            if ns and ns.loaded:
                try:
                    pred_priority, pred_delay = ns.predict(user_input)
                except Exception as exc:
                    logger.warning("Neural scheduler prediction failed: %s", exc)

            # Fallback/hybrid delay logic: use explicit regex extracted delay if present
            regex_delay = extract_goal_delay_seconds(user_input)
            delay_seconds = regex_delay if regex_delay > 0.0 else pred_delay

            goal_id = goal_manager.create_goal(
                description=description,
                priority=pred_priority,
            )
            try:
                goal_manager.start_goal(goal_id)
            except (ValueError, KeyError) as exc:
                logger.warning("Failed to start goal %r: %s", goal_id, exc)
            scheduler.enqueue(
                mission_id=goal_id,
                goal_id=goal_id,
                delay_seconds=delay_seconds,
                description=description,
            )
            
            # Nicely format response mentioning the predicted parameters
            if delay_seconds > 0.0:
                response = f"Goal set: {description} (priority: {pred_priority}, scheduled in {int(delay_seconds)}s)"
            else:
                response = f"Goal set: {description} (priority: {pred_priority}, scheduled immediately)"
            
            return GoalIntentResult(
                response=response,
                mutated=True,
            )

    if any(keyword in text for keyword in _GOAL_LIST_KEYWORDS):
        goals = goal_manager.active_goals()
        if not goals:
            return GoalIntentResult(response="No active goals.")
        lines = [f"- [{goal.priority}] {goal.description}" for goal in goals]
        return GoalIntentResult(response="Active goals:\n" + "\n".join(lines))

    return None


def handle_preference_intent(
    text: str,
    user_input: str,
    *,
    memory: Any,
) -> str | None:
    if text.startswith("remember i like "):
        value = user_input[16:].strip()
        if value:
            memory.store_preference(f"likes_{value[:12]}", value)
            return f"I will remember you like {value}."

    if text.startswith("my name is "):
        value = user_input[11:].strip()
        if value:
            memory.store_preference("name", value)
            return f"I will remember your name is {value}."

    if text.startswith("i prefer "):
        value = user_input[9:].strip()
        if value:
            memory.store_preference(f"prefer_{value[:12]}", value)
            return f"I will remember you prefer {value}."

    if text.startswith("i work in "):
        value = user_input[10:].strip()
        if value:
            memory.store_preference("work", value)
            return f"I will remember you work in {value}."

    return None


def extract_goal_delay_seconds(user_input: str) -> float:
    lowered = user_input.lower()
    if "tomorrow" in lowered:
        return 24 * 60 * 60
    match = re.search(
        r"\bin\s+(\d+)\s+(minute|minutes|hour|hours|day|days)\b",
        lowered,
    )
    if not match:
        return 0.0
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("minute"):
        return float(value * 60)
    if unit.startswith("hour"):
        return float(value * 60 * 60)
    if unit.startswith("day"):
        return float(value * 24 * 60 * 60)
    return 0.0


__all__ = [
    "GoalIntentResult",
    "extract_goal_delay_seconds",
    "handle_goal_intent",
    "handle_preference_intent",
]
