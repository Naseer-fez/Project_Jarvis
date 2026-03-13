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
from core.desktop_actions import handle_desktop_command
from core.llm.model_router import ModelRouter
from core.llm.client import LLMClientV2
from core.memory.hybrid_memory import HybridMemory
from core.profile import UserProfileEngine
from core.proactive.background_monitor import BackgroundMonitor
from core.proactive.notifier import NotificationManager
from core.synthesis import ProfileSynthesizer
from core.agent.agent_loop import AgentLoopEngine
from core.agent.state_machine import StateMachine
from core.llm.task_planner import TaskPlanner
from core.tools.tool_router import ToolRouter
from core.autonomy.risk_evaluator import RiskEvaluator
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.tools.builtin_tools import register_all_tools

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
        self.profile = UserProfileEngine()
        
        base_url = "http://localhost:11434"
        if isinstance(self.config, configparser.ConfigParser) and self.config.has_section("ollama"):
            base_url = self.config.get("ollama", "base_url", fallback=base_url)

        self.llm = LLMClientV2(
            hybrid_memory=self.memory,
            model=self.model_router.route("chat"),
            profile=self.profile,
            base_url=base_url,
        )
        self.llm.set_router(self.model_router)
        self.synthesizer = ProfileSynthesizer(self.llm)
        self._conversation_buffer: list[str] = []
        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._on_state_update = lambda **_: None
        try:
            setattr(self.llm, "profile", self.profile)
        except Exception:  # noqa: BLE001
            pass
        self.exchanges = 0

        self.state_machine = StateMachine()
        self.task_planner = TaskPlanner(self.config)
        self.tool_router = ToolRouter()
        register_all_tools(self.tool_router)
        self.risk_evaluator = RiskEvaluator(self.config)
        self.autonomy_governor = AutonomyGovernor(level=3)
        self.agent_loop = AgentLoopEngine(
            state_machine=self.state_machine,
            task_planner=self.task_planner,
            tool_router=self.tool_router,
            risk_evaluator=self.risk_evaluator,
            autonomy_governor=self.autonomy_governor,
            model=model_name,
            ollama_url=base_url,
            llm=self.llm,
        )

        self.voice_loop = None
        self.goal_manager = GoalManager()
        self.scheduler = Scheduler()
        self._goal_check_task: asyncio.Task | None = None
        self._goal_check_interval_seconds = 300
        if isinstance(config, configparser.ConfigParser):
            minutes = config.getint("proactive", "goal_check_interval_minutes", fallback=5)
            self._goal_check_interval_seconds = max(1, minutes) * 60
        self._goals_file = Path(
            config.get("memory", "goals_file", fallback="memory/goals.json")
            if isinstance(config, configparser.ConfigParser)
            else "memory/goals.json"
        )
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

    async def _handle_goal_intent(self, text: str, user_input: str) -> str | None:
        if any(kw in text for kw in ("remind me", "set goal", "schedule", "don't forget", "remember to")):
            description = user_input
            for kw in ("remind me to", "set goal", "schedule", "don't forget to", "remember to"):
                description = re.sub(re.escape(kw), "", description, flags=re.IGNORECASE).strip()
            description = description.strip(" .?!")
            if description:
                goal_id = self.goal_manager.create_goal(description=description)
                try:
                    self.goal_manager.start_goal(goal_id)
                except (ValueError, KeyError) as exc:
                    logger.warning("Failed to start goal %r: %s", goal_id, exc)
                delay_seconds = self._extract_goal_delay_seconds(user_input)
                self.scheduler.enqueue(
                    mission_id=goal_id,
                    goal_id=goal_id,
                    delay_seconds=delay_seconds,
                    description=description,
                )
                self._persist_goal_state()
                self._dashboard_update(active_goals=len(self.goal_manager.active_goals()))
                return f"✓ Goal set: {description}"
        
        if any(kw in text for kw in ("what are my goals", "show goals", "list goals", "my goals")):
            goals = self.goal_manager.active_goals()
            if not goals:
                return "No active goals."
            lines = [f"- [{g.priority}] {g.description}" for g in goals]
            return "Active goals:\n" + "\n".join(lines)
            
        return None

    async def _handle_preference_intent(self, text: str, user_input: str) -> str | None:
        if text.startswith("remember i like "):
            value = user_input[16:].strip()
            if value:
                self.memory.store_preference(f"likes_{value[:12]}", value)
                return f"I will remember you like {value}."

        if text.startswith("my name is "):
            value = user_input[11:].strip()
            if value:
                self.memory.store_preference("name", value)
                return f"I will remember your name is {value}."

        if text.startswith("i prefer "):
            value = user_input[9:].strip()
            if value:
                self.memory.store_preference(f"prefer_{value[:12]}", value)
                return f"I will remember you prefer {value}."

        if text.startswith("i work in "):
            value = user_input[10:].strip()
            if value:
                self.memory.store_preference("work", value)
                return f"I will remember you work in {value}."
                
        return None

    async def _dispatch_llm(self, text: str, trace_id: str) -> str:
        context = self.memory.build_context_block(text)
        profile_injection = self.profile.get_system_prompt_injection()
        style_instruction = self.profile.get_communication_style()
        profile_guidance = f"{profile_injection}\n{style_instruction}".strip()

        selected_model = self.model_router.get_best_available("chat")
        self.llm.model = selected_model
        self.llm.model_name = selected_model

        messages = [{"role": "user", "content": text}]
        profile_summary = self.profile.get_communication_style() if self.profile else ""
        
        logger.info("[trace=%s] Dispatching to LLM: %r", trace_id, selected_model)
        response = await self.llm.chat_async(
            messages, 
            query_for_memory=text, 
            profile_summary=profile_summary,
            trace_id=trace_id
        )

        if not response or response == "LLM Offline.":
            prefs = self.memory.recall_preferences(text, top_k=1)
            if prefs and prefs[0].get("value"):
                return f"Offline fallback from memory: {prefs[0]['value']}"
            return "I don't know while offline."

        self.memory.store_conversation(text, response, self.session_id)
        return response

    async def process(self, user_input: str, trace_id: str | None = None) -> str:
        if not trace_id:
            trace_id = uuid.uuid4().hex[:8]
        logger.info("[trace=%s] Controller process started", trace_id)
        
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
            return _respond(
                "Commands: status, help, exit, remember <fact>, what's <query>, "
                "open <app>, search <query> in <browser>"
            )

        if (goal_res := await self._handle_goal_intent(lowered, text)):
            return _respond(goal_res)
            
        if (pref_res := await self._handle_preference_intent(lowered, text)):
            return _respond(pref_res)

        if (desktop_res := await handle_desktop_command(text)):
            return _respond(desktop_res)

        # Trigger AgentLoopEngine if the query implies research or action
        agent_keywords = ["search", "look up", "find", "check", "scrape", "get", "download", "fetch", "read", "click", "analyze"]
        if len(text.split()) > 2 and any(kw in lowered for kw in agent_keywords):
            self._dashboard_update(state="PLANNING", last_input=user_input)
            plan = await asyncio.to_thread(self.task_planner.plan, text)
            if plan and plan.get("tools_required"):
                self._dashboard_update(state="EXECUTE", last_input=user_input)
                trace = await self.agent_loop.run(goal=text, context=self.memory.build_context_block(text))
                return _respond(trace.final_response)

        response = await self._dispatch_llm(text, trace_id=trace_id)
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

            response = await self.process(text)
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
            except Exception:
                logger.warning("VoiceLayer stop error", exc_info=True)

        if self.voice_loop is not None:
            try:
                await self.voice_loop.stop()
            except Exception:
                logger.warning("voice_loop stop error", exc_info=True)

    async def _voice_text_handler(self, text: str) -> str:
        """Called by VoiceLayer when speech is recognized."""
        self._dashboard_update(state="THINKING", last_input=text)
        response = await self.process(text)
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
            except Exception:
                logger.warning("Goal check loop error", exc_info=True)

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
            self._on_state_update(**kwargs)
        except Exception:
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
