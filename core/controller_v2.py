"""JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes."""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.agentic.scheduler import Scheduler
from core.autonomy.goal_manager import GoalManager
from core.llm.model_router import ModelRouter
from core.llm.llm_v2 import LLMClientV2
from core.memory.hybrid_memory import HybridMemory
from core.profile import UserProfileEngine
from core.proactive.background_monitor import BackgroundMonitor
from core.proactive.notifier import NotificationManager
from core.synthesis import ProfileSynthesizer

logger = logging.getLogger(__name__)


class JarvisControllerV2:
    def __init__(
        self,
        config: configparser.ConfigParser | None = None,
        voice: bool = False,
        db_path: str = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.config = config if isinstance(config, configparser.ConfigParser) else configparser.ConfigParser()
        self.voice_enabled = bool(voice)

        if isinstance(config, configparser.ConfigParser):
            db_path = config.get("memory", "db_path", fallback=config.get("memory", "sqlite_file", fallback=db_path))
            chroma_path = config.get("memory", "chroma_path", fallback=config.get("memory", "chroma_dir", fallback=chroma_path))
            model_name = config.get(
                "models",
                "chat_model",
                fallback=config.get("llm", "model", fallback=config.get("ollama", "planner_model", fallback=model_name)),
            )
            embedding_model = config.get("memory", "embedding_model", fallback=embedding_model)

        self.session_id = uuid.uuid4().hex[:8]
        self.memory = HybridMemory(db_path, chroma_path=chroma_path, model_name=embedding_model)
        self.model_router = ModelRouter(config=self.config)
        self.llm = LLMClientV2(model_name=self.model_router.route("chat"))
        self.profile = UserProfileEngine()
        self.synthesizer = ProfileSynthesizer(self.llm)
        self._conversation_buffer: list[str] = []
        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        try:
            setattr(self.llm, "profile", self.profile)
        except Exception:  # noqa: BLE001
            pass
        self.exchanges = 0

        self.voice_loop = None
        self.goal_manager = GoalManager()
        self.scheduler = Scheduler()
        self._goal_check_task: asyncio.Task | None = None
        self._goal_check_interval_seconds = 300
        if isinstance(config, configparser.ConfigParser):
            minutes = config.getint("proactive", "goal_check_interval_minutes", fallback=5)
            self._goal_check_interval_seconds = max(1, minutes) * 60
        self._goals_file = Path("memory/goals.json")
        self._load_goal_state()
        self.notifier = NotificationManager()
        self.monitor = BackgroundMonitor(self.notifier, self.config)

        self.voice_enabled = voice
        self._voice_layer = None
        if voice:
            try:
                from core.voice.voice_layer import VoiceLayer

                self._voice_layer = VoiceLayer(
                    controller=self,
                    config=self.config,
                )
                logger.info("VoiceLayer initialized")
            except ImportError as e:
                logger.warning(f"Voice unavailable: {e}")
            except Exception as e:
                logger.warning(f"VoiceLayer init failed: {e}")

    def initialize(self) -> dict[str, Any]:
        index_path = ""
        if isinstance(self.config, configparser.ConfigParser):
            index_path = self.config.get("memory", "index_path", fallback="")

        memory_status = self.memory.initialize(index_path=index_path)
        return {
            "session_id": self.session_id,
            "memory_mode": memory_status.get("mode"),
            "memory_init": memory_status,
        }

    def process(self, user_input: str) -> str:
        self._dashboard_update(state="THINKING", last_input=user_input)
        self.exchanges += 1
        text = (user_input or "").strip()
        lowered = text.lower()

        def _respond(response: str) -> str:
            try:
                self.profile.update_from_conversation(user_input, response)
                self._conversation_buffer.append(f"User: {user_input}\nJarvis: {response}")

                if self.synthesizer.should_run(self.profile):
                    self._schedule_synthesis(self._conversation_buffer[-20:])
                    self._conversation_buffer.clear()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Profile update/synthesis scheduling failed: %s", exc)

            self._dashboard_update(
                state="IDLE",
                last_response=response,
                active_goals=len(self.goal_manager.active_goals()),
            )
            return response

        if lowered == "status":
            return _respond(f"Session: {self.session_id} | Memory Mode: {self.memory.mode}")
        if lowered == "help":
            return _respond("Commands: status, help, exit, remember <fact>, what's <query>")

        if any(kw in lowered for kw in ("remind me", "set goal", "schedule", "don't forget", "remember to")):
            description = text
            for kw in ("remind me to", "set goal", "schedule", "don't forget to", "remember to"):
                description = re.sub(re.escape(kw), "", description, flags=re.IGNORECASE).strip()
            description = description.strip(" .?!")
            if description:
                goal_id = self.goal_manager.create_goal(description=description)
                try:
                    self.goal_manager.start_goal(goal_id)
                except Exception:
                    pass
                delay_seconds = self._extract_goal_delay_seconds(text)
                self.scheduler.enqueue(
                    mission_id=goal_id,
                    goal_id=goal_id,
                    delay_seconds=delay_seconds,
                    description=description,
                )
                self._persist_goal_state()
                self._dashboard_update(active_goals=len(self.goal_manager.active_goals()))
                return _respond(f"✓ Goal set: {description}")

        if any(kw in lowered for kw in ("what are my goals", "show goals", "list goals", "my goals")):
            goals = self.goal_manager.active_goals()
            if not goals:
                return _respond("No active goals.")
            lines = [f"- [{g.priority}] {g.description}" for g in goals]
            return _respond("Active goals:\n" + "\n".join(lines))

        if lowered.startswith("remember i like "):
            value = text[16:].strip()
            if value:
                self.memory.store_preference(f"likes_{value[:12]}", value)
                return _respond(f"I will remember you like {value}.")

        if lowered.startswith("my name is "):
            value = text[11:].strip()
            if value:
                self.memory.store_preference("name", value)
                return _respond(f"I will remember your name is {value}.")

        if lowered.startswith("i prefer "):
            value = text[9:].strip()
            if value:
                self.memory.store_preference(f"prefer_{value[:12]}", value)
                return _respond(f"I will remember you prefer {value}.")

        if lowered.startswith("i work in "):
            value = text[10:].strip()
            if value:
                self.memory.store_preference("work", value)
                return _respond(f"I will remember you work in {value}.")

        context = self.memory.build_context_block(text)
        profile_injection = self.profile.get_system_prompt_injection()
        style_instruction = self.profile.get_communication_style()
        profile_guidance = f"{profile_injection}\n{style_instruction}".strip()

        system_context = context
        if profile_guidance:
            system_context = f"{context}\n\nPROFILE GUIDANCE:\n{profile_guidance}".strip()

        self.llm.model_name = self.model_router.get_best_available("chat")
        response = self.llm.chat(text, system_context=system_context)

        if response == "LLM Offline.":
            prefs = self.memory.recall_preferences(text, top_k=1)
            if prefs and prefs[0].get("value"):
                return _respond(f"Offline fallback from memory: {prefs[0]['value']}")
            return _respond("I don't know while offline.")

        self.memory.store_conversation(text, response, self.session_id)
        return _respond(response)

    def session_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exchanges": self.exchanges,
        }

    async def start(self) -> None:
        self._runtime_loop = asyncio.get_running_loop()
        self.initialize()
        asyncio.create_task(self.monitor.start(), name="jarvis_resource_monitor_start")

    async def run_cli(self) -> None:
        """Run either standard CLI or voice loop based on --voice/config."""
        if self._goal_check_task is None or self._goal_check_task.done():
            self._goal_check_task = asyncio.create_task(self._check_due_goals(), name="jarvis_goal_due_checker")
        asyncio.create_task(self.monitor.start(), name="jarvis_resource_monitor_start_cli")

        if self._voice_layer is not None:
            logger.info("Starting in voice mode...")
            self._dashboard_update(state="IDLE")
            await self._voice_layer.start()
            task = getattr(self._voice_layer, "_task", None)
            if task is not None:
                await task
            return

        print(f"Jarvis V2 ready (session {self.session_id}). Type 'exit' to quit.")
        loop = asyncio.get_running_loop()
        self._runtime_loop = loop

        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "You: ")
            except EOFError:
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break

            response = await loop.run_in_executor(None, self.process, text)
            print(f"Jarvis: {response}")

    async def shutdown(self) -> None:
        await self.monitor.stop()

        if self._goal_check_task is not None and not self._goal_check_task.done():
            self._goal_check_task.cancel()
            try:
                await self._goal_check_task
            except asyncio.CancelledError:
                pass

        if self._voice_layer is not None:
            try:
                await self._voice_layer.stop()
            except Exception as e:
                logger.warning(f"VoiceLayer stop error: {e}")

        if self.voice_loop is not None:
            await self.voice_loop.stop()

    async def _voice_text_handler(self, text: str) -> str:
        """Called by VoiceLayer when speech is recognized."""
        self._dashboard_update(state="THINKING", last_input=text)
        response = self.process(text)
        self._dashboard_update(state="IDLE", last_response=response)
        return response

    async def _check_due_goals(self) -> None:
        """Background task: check for overdue goals every 5 minutes."""
        while True:
            await asyncio.sleep(self._goal_check_interval_seconds)
            try:
                due_items = self.scheduler.due()
                for item in due_items:
                    msg = f"Due: {item.description or item.goal_id}"
                    self.notifier.notify(msg)
                    item.mark_completed()
                    if self._voice_layer is not None:
                        try:
                            await self._speak_via_voice_layer(msg)
                        except Exception:
                            pass
                if due_items:
                    self._persist_goal_state()
                self._dashboard_update(active_goals=len(self.goal_manager.active_goals()))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Goal check error: {e}")

    async def _speak_via_voice_layer(self, text: str) -> None:
        voice_loop = getattr(self._voice_layer, "_loop", None)
        tts = getattr(voice_loop, "tts", None)
        speak = getattr(tts, "speak", None)
        if speak is None:
            return
        result = speak(text)
        if asyncio.iscoroutine(result):
            await result

    def _schedule_synthesis(self, conversations: list[str]) -> None:
        coro = self.synthesizer.synthesize(conversations, self.profile)
        try:
            asyncio.create_task(coro)
            return
        except RuntimeError:
            pass

        if self._runtime_loop is not None and self._runtime_loop.is_running():
            self._runtime_loop.call_soon_threadsafe(asyncio.create_task, coro)
            return

        coro.close()
        logger.warning("No running loop available; skipped profile synthesis task.")

    def _extract_goal_delay_seconds(self, user_input: str) -> float:
        lowered = user_input.lower()
        if "tomorrow" in lowered:
            return 24 * 60 * 60
        match = re.search(r"\bin\s+(\d+)\s+(minute|minutes|hour|hours|day|days)\b", lowered)
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

    def _dashboard_update(self, **kwargs: Any) -> None:
        try:
            from dashboard.server import update_state

            update_state(**kwargs)
        except ImportError:
            pass

    def _persist_goal_state(self) -> None:
        try:
            self._goals_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "goals": self.goal_manager.snapshot(),
                "schedule": self.scheduler.snapshot(),
            }
            self._goals_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist goals: %s", exc)

    def _load_goal_state(self) -> None:
        if not self._goals_file.exists():
            return
        try:
            data = json.loads(self._goals_file.read_text(encoding="utf-8"))
            goals = data.get("goals", [])
            schedule = data.get("schedule", [])
            if isinstance(goals, list):
                self.goal_manager.restore(goals)
            if isinstance(schedule, list):
                self.scheduler.restore(schedule)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load goals: %s", exc)


# Backward-compatible alias used by main.py
Controller = JarvisControllerV2

__all__ = ["JarvisControllerV2", "Controller"]
