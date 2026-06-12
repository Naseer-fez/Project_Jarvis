# API Analyst Report: controller\goal_runner.py

## Dependencies
- `import asyncio`
- `import json`
- `import logging`
- `from datetime import datetime`
- `from datetime import timezone`
- `from pathlib import Path`
- `from typing import Callable`

## Schemas & API Contracts (Classes)

### Class `GoalRunner`
> Handles goal persistence, checking due goals, and notifications.

**Methods:**
- `def __init__(self, goal_manager, scheduler, notifier, voice_layer, goals_file: Path, goal_check_interval_seconds: int, dashboard_update_cb: Callable)`
- `async def load_goal_state(self) -> None`
- `async def persist_goal_state(self) -> None`
- `async def speak_via_voice_layer(self, text: str) -> None`
- `async def check_due_goals(self) -> None`

