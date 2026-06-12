# API Analyst Report: controller\intents.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `import re`
- `from dataclasses import dataclass`
- `from typing import Any`

## Configuration Variables
- `_GOAL_CREATE_KEYWORDS` = `('remind me', 'set goal', 'schedule', "don't forget", 'remember to')`
- `_GOAL_STRIP_KEYWORDS` = `('remind me to', 'set goal', 'schedule', "don't forget to", 'remember to')`
- `_GOAL_LIST_KEYWORDS` = `('what are my goals', 'show goals', 'list goals', 'my goals')`

## Schemas & API Contracts (Classes)

### Class `GoalIntentResult`
**Fields/Schema:**
  - `response: str`
  - `mutated: bool`



## Functions & Endpoints

### `parse_time_delay_with_parsedatetime`
`def parse_time_delay_with_parsedatetime(text: str) -> float`
### `handle_goal_intent`
`def handle_goal_intent(text: str, user_input: str, *, goal_manager: Any, scheduler: Any) -> GoalIntentResult | None`
### `handle_preference_intent`
`async def handle_preference_intent(text: str, user_input: str, *, memory: Any) -> str | None`
### `extract_goal_delay_seconds`
`def extract_goal_delay_seconds(user_input: str) -> float`