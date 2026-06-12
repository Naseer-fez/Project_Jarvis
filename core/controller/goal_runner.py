import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

class GoalRunner:
    """Handles goal persistence, checking due goals, and notifications."""
    
    def __init__(
        self,
        goal_manager,
        scheduler,
        notifier,
        voice_layer,
        goals_file: Path,
        goal_check_interval_seconds: int,
        dashboard_update_cb: Callable
    ):
        self.goal_manager = goal_manager
        self.scheduler = scheduler
        self.notifier = notifier
        self.voice_layer = voice_layer
        self.goals_file = Path(goals_file)
        self.goal_check_interval_seconds = goal_check_interval_seconds
        self.dashboard_update = dashboard_update_cb

    async def load_goal_state(self) -> None:
        if not self.goals_file.exists():
            return
        try:
            def _read():
                return self.goals_file.read_text(encoding="utf-8")
            content = await asyncio.to_thread(_read)
            data = json.loads(content)
            goals = data.get("goals", [])
            schedule = data.get("schedule", [])
            if isinstance(goals, list):
                self.goal_manager.restore(goals)
            if isinstance(schedule, list):
                self.scheduler.restore(schedule)
        except Exception as exc:
            logger.warning("Failed to load goals: %s", exc, exc_info=True)

    async def persist_goal_state(self) -> None:
        try:
            self.goals_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "goals": self.goal_manager.snapshot(),
                "schedule": self.scheduler.snapshot(),
            }
            def _write():
                self.goals_file.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
            await asyncio.to_thread(_write)
        except Exception as exc:
            logger.warning("Failed to persist goals: %s", exc, exc_info=True)

    async def speak_via_voice_layer(self, text: str) -> None:
        if self.voice_layer is None:
            return
        voice_loop = getattr(self.voice_layer, "_loop", None)
        tts = getattr(voice_loop, "tts", None)
        speak = getattr(tts, "speak", None)
        if speak is None:
            return
        result = speak(text)
        if asyncio.iscoroutine(result):
            await result

    async def check_due_goals(self) -> None:
        backoff = 1.0
        while True:
            try:
                await asyncio.sleep(self.goal_check_interval_seconds)
                due_items = self.scheduler.due()
                for item in due_items:
                    msg = f"Due: {item.description or item.goal_id}"
                    self.notifier.notify(msg)
                    item.mark_completed()
                    try:
                        await self.speak_via_voice_layer(msg)
                    except Exception as e:
                        logger.warning("Failed to speak due goal via voice layer: %s", e, exc_info=True)
                if due_items:
                    await self.persist_goal_state()
                self.dashboard_update(active_goals=len(self.goal_manager.active_goals()))
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Goal check loop error: %s", e, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
