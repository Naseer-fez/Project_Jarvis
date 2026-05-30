"""JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes."""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from core.agent.agent_loop import AgentLoopEngine
from core.state_machine import StateMachine
from core.agentic.scheduler import Scheduler
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.goal_manager import GoalManager
from core.autonomy.risk_evaluator import RiskEvaluator
from core.controller.intents import (
    extract_goal_delay_seconds,
    handle_goal_intent,
    handle_preference_intent,
)
from core.controller.services import build_controller_services
from core.desktop.shortcuts import handle_desktop_command, plan_desktop_command
from core.llm.client import LLMClientV2
from core.llm.defaults import DEFAULT_MODEL
from core.llm.model_router import ModelRouter
from core.planner.planner import TaskPlanner
from core.memory.hybrid_memory import HybridMemory
from core.profile import UserProfileEngine
from core.proactive.background_monitor import BackgroundMonitor
from core.proactive.notifier import NotificationManager
from core.synthesis import ProfileSynthesizer
from core.tools.builtin_tools import register_all_tools
from core.registry.registry import CapabilityRegistry

logger = logging.getLogger(__name__)

_DESKTOP_CONTROL_KEYWORDS = (
    "mouse",
    "cursor",
    "desktop",
    "screen",
    "keyboard",
    "hotkey",
    "click",
)
_AGENTIC_KEYWORDS = (
    "search",
    "look up",
    "find",
    "check",
    "scrape",
    "get",
    "download",
    "fetch",
    "read",
    "analyze",
    "mouse",
    "cursor",
    "desktop",
    "screen",
    "keyboard",
    "hotkey",
    "click",
    # Web / research triggers
    "browse",
    "internet",
    "online",
    "web",
    "website",
    "latest",
    "current",
    "news",
    "price",
    "weather",
    "what is",
    "what are",
    "who is",
    "how to",
    "when did",
    "where is",
    "tell me about",
    "look it up",
    "google",
    # System / File tools keywords
    "sort",
    "organize",
    "list",
    "write",
    "delete",
    "create",
    "copy",
    "move",
    "run",
    "execute",
    "launch",
    "open",
    "command",
)

# Phrases that unambiguously indicate the user wants a live web search.
# When any of these are detected Jarvis skips the LLM planner entirely
# and calls web_search directly, then synthesises the answer.
_WEB_SEARCH_EXPLICIT_PHRASES = (
    "search the web",
    "search the internet",
    "search online",
    "browse the web",
    "browse the internet",
    "browse online",
    "look it up online",
    "look up online",
    "google it",
    "google for",
    "find online",
    "find on the internet",
    "find on the web",
    "search for",
    "search web for",
    "web search",
    "internet search",
    "use internet",
    "use the internet",
)


class JarvisControllerV2:
    def __init__(
        self,
        config: configparser.ConfigParser | None = None,
        voice: bool = False,
        db_path: str = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = "all-MiniLM-L6-v2",
        container: Any = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, configparser.ConfigParser)
            else configparser.ConfigParser()
        )
        self.voice_enabled = bool(voice)
        self.session_id = uuid.uuid4().hex[:8]
        self._gui_automation_enabled = self.config.getboolean(
            "execution",
            "allow_gui_automation",
            fallback=False,
        )
        self._app_launch_enabled = self.config.getboolean(
            "execution",
            "allow_app_launch",
            fallback=True,
        )

        settings, services = build_controller_services(
            self.config,
            container=container,
            db_path=db_path,
            chroma_path=chroma_path,
            model_name=model_name,
            embedding_model=embedding_model,
            memory_cls=HybridMemory,
            model_router_cls=ModelRouter,
            profile_cls=UserProfileEngine,
            llm_cls=LLMClientV2,
            synthesizer_cls=ProfileSynthesizer,
            state_machine_cls=StateMachine,
            task_planner_cls=TaskPlanner,
            tool_router_cls=CapabilityRegistry,
            risk_evaluator_cls=RiskEvaluator,
            autonomy_governor_cls=AutonomyGovernor,
            agent_loop_cls=AgentLoopEngine,
            goal_manager_cls=GoalManager,
            scheduler_cls=Scheduler,
            notifier_cls=NotificationManager,
            monitor_cls=BackgroundMonitor,
            register_tools=register_all_tools,
        )

        self.memory = services.memory
        self.model_router = services.model_router
        self.profile = services.profile
        self.llm = services.llm
        self.synthesizer = services.synthesizer
        self.state_machine = services.state_machine
        self.task_planner = services.task_planner
        self.tool_router = services.tool_router
        self.risk_evaluator = services.risk_evaluator
        self.autonomy_governor = services.autonomy_governor
        self.agent_loop = services.agent_loop
        self.goal_manager = services.goal_manager
        self.scheduler = services.scheduler
        self.notifier = services.notifier
        self.monitor = services.monitor
        self.desktop_executor = services.desktop_executor
        self.desktop_observer = services.desktop_observer
        self.desktop_bridge = services.desktop_bridge
        self.container = services.container

        self._conversation_buffer: list[str] = []
        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._on_state_update = lambda **_: None
        self.exchanges = 0
        self.voice_loop = None
        self._goal_check_task: asyncio.Task | None = None
        self._goal_check_interval_seconds = settings.goal_check_interval_seconds
        self._goals_file = settings.goals_file
        self._load_goal_state()

        self.live_automation = None
        if self.config.has_section("automation") and self.config.getboolean("automation", "enabled", fallback=False):
            try:
                from core.automation.live_automation import LiveAutomationEngine

                async def command_handler(cmd: str) -> str:
                    return await self.process(cmd)

                self.live_automation = LiveAutomationEngine(
                    config=self.config,
                    memory=self.memory,
                    llm=self.llm,
                    command_handler=command_handler,
                    desktop_observer=services.desktop_observer,
                    notifier=self.notifier,
                )
            except Exception as exc:
                logger.warning("Failed to initialize LiveAutomationEngine: %s", exc)

        self._voice_layer = None
        if voice:
            try:
                from core.voice.voice_layer import VoiceLayer

                self._voice_layer = VoiceLayer(
                    controller=self,
                    config=self.config,
                )
                logger.info("VoiceLayer initialized")
            except ImportError as exc:
                logger.warning("Voice unavailable: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VoiceLayer init failed: %s", exc)

    async def initialize(self) -> dict[str, Any]:
        index_path = ""
        if isinstance(self.config, configparser.ConfigParser):
            index_path = self.config.get("memory", "index_path", fallback="")

        memory_status = await self.memory.initialize(index_path=index_path)
        return {
            "session_id": self.session_id,
            "memory_mode": memory_status.get("mode"),
            "memory_init": memory_status,
        }

    async def _handle_goal_intent(self, text: str, user_input: str) -> str | None:
        result = handle_goal_intent(
            text,
            user_input,
            goal_manager=self.goal_manager,
            scheduler=self.scheduler,
        )
        if result is None:
            return None
        if result.mutated:
            self._persist_goal_state()
            self._dashboard_update(active_goals=len(self.goal_manager.active_goals()))
        return result.response

    async def _handle_preference_intent(
        self,
        text: str,
        user_input: str,
    ) -> str | None:
        return await handle_preference_intent(
            text,
            user_input,
            memory=self.memory,
        )

    async def _dispatch_llm(self, text: str, trace_id: str) -> str:
        await self.memory.build_context_block(text)
        profile_injection = self.profile.get_system_prompt_injection()
        style_instruction = self.profile.get_communication_style()
        _ = f"{profile_injection}\n{style_instruction}".strip()

        selected_model = self.model_router.get_best_available("chat")
        self.llm.model = selected_model

        messages = [{"role": "user", "content": text}]
        profile_summary = (
            self.profile.get_communication_style() if self.profile else ""
        )

        logger.info("[trace=%s] Dispatching to LLM: %r", trace_id, selected_model)
        response = await self.llm.chat_async(
            messages,
            query_for_memory=text,
            profile_summary=profile_summary,
            trace_id=trace_id,
        )

        if not response or response == "LLM Offline.":
            prefs = await self.memory.recall_preferences(text, top_k=1)
            if prefs and prefs[0].get("value"):
                return f"Offline fallback from memory: {prefs[0]['value']}"
            return "I don't know while offline."

        await self.memory.store_conversation(text, response, self.session_id)
        return response

    def _looks_like_desktop_control_request(self, lowered: str) -> bool:
        return any(keyword in lowered for keyword in _DESKTOP_CONTROL_KEYWORDS)

    def _desktop_control_disabled_message(self) -> str:
        return (
            "Desktop control is disabled. Set [execution] allow_gui_automation = true "
            "in config/jarvis.ini, install the desktop extras with "
            "'pip install -r requirements/desktop.txt', then restart Jarvis."
        )

    def _app_launch_disabled_message(self) -> str:
        return (
            "Application launch is disabled. Set [execution] allow_app_launch = true "
            "in config/jarvis.ini, then restart Jarvis."
        )

    def _is_explicit_web_search(self, lowered: str) -> bool:
        """Return True when the user unambiguously asks for a live web search."""
        return any(phrase in lowered for phrase in _WEB_SEARCH_EXPLICIT_PHRASES)

    async def _handle_web_search(self, user_input: str, trace_id: str) -> str:
        """Perform a live web search and synthesise a natural-language reply."""
        try:
            from core.tools.web_tools import web_search as _web_search

            # Extract a clean query from the raw user input via basic cleanup
            from core.tools.web_tools import _basic_query_cleanup
            query = _basic_query_cleanup(user_input)
            if not query:
                query = user_input.strip()

            logger.info("[trace=%s] Explicit web search: %r", trace_id, query)
            raw_results = await _web_search(query, max_results=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[trace=%s] Web search failed: %s", trace_id, exc)
            return f"Web search failed: {exc}"

        if not raw_results or raw_results.startswith("Web search is disabled"):
            # Search didn't work — fall back to LLM
            return await self._dispatch_llm(user_input, trace_id=trace_id)

        if raw_results.startswith("Search failed"):
            return raw_results

        # Ask the LLM to turn the raw search output into a natural answer
        synthesis_prompt = (
            f"The user asked: {user_input}\n\n"
            f"Here are the live web search results:\n{raw_results}\n\n"
            "Please give a concise, helpful reply based only on these results. "
            "Cite URLs where relevant. Do not invent information not present in the results."
        )
        await self.memory.build_context_block(user_input)
        selected_model = self.model_router.get_best_available("chat")
        self.llm.model = selected_model
        messages = [{"role": "user", "content": synthesis_prompt}]
        try:
            response = await self.llm.chat_async(
                messages,
                query_for_memory=user_input,
                profile_summary=self.profile.get_communication_style() if self.profile else "",
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[trace=%s] Web search synthesis LLM call failed: %s", trace_id, exc)
            # Return parsed raw results if synthesis fails
            return self._format_raw_fallback(raw_results)

        if not response or response == "LLM Offline.":
            return self._format_raw_fallback(raw_results)

        await self.memory.store_conversation(user_input, response, self.session_id)
        return response

    def _format_raw_fallback(self, raw_results: str) -> str:
        """Parse and format raw search results nicely when LLM synthesis is not available."""
        import re
        lines = raw_results.splitlines()
        formatted_lines = []
        current_title = None
        current_num = None

        for line in lines:
            line_stripped = line.strip()
            # Match "1. Title"
            match = re.match(r"^(\d+)\.\s+(.*)$", line_stripped)
            if match:
                if current_title is not None:
                    formatted_lines.append(f"{current_num}. {current_title}")
                current_num = match.group(1)
                current_title = match.group(2)
                continue

            # Match "URL: https://..."
            if line_stripped.startswith("URL:") and current_title is not None:
                url = line_stripped[4:].strip()
                formatted_lines.append(f"{current_num}. {current_title} ({url})")
                current_title = None
                current_num = None
                continue

            if current_title is not None:
                formatted_lines.append(f"{current_num}. {current_title}")
                current_title = None
                current_num = None

            formatted_lines.append(line)

        if current_title is not None:
            formatted_lines.append(f"{current_num}. {current_title}")

        return "\n".join(formatted_lines)

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
                self._conversation_buffer.append(
                    f"User: {user_input}\nJarvis: {response}"
                )

                if self.synthesizer.should_run(self.profile):
                    self._schedule_synthesis(self._conversation_buffer[-20:])
                    self._conversation_buffer.clear()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Profile update/synthesis scheduling failed: %s",
                    exc,
                )

            self._dashboard_update(
                state="IDLE",
                last_response=response,
                active_goals=len(self.goal_manager.active_goals()),
            )
            return response

        if lowered == "status":
            return _respond(
                f"Session: {self.session_id} | Memory Mode: {self.memory.mode}"
            )
        if lowered == "help":
            return _respond(
                "Commands: status, help, exit, remember <fact>, what's <query>, "
                "open <app>, search <query> in <browser>"
            )

        # Check for live automation commands
        if hasattr(self, "live_automation") and self.live_automation is not None:
            if lowered == "automation status":
                status_info = self.live_automation.status()
                response = f"{self.live_automation.status_line()}\nDrop Root: {status_info.get('drop_root')}\nCommands Dir: {status_info.get('commands_dir')}\nRAG Dir: {status_info.get('rag_dir')}"
                return _respond(response)
            elif lowered == "automation scan":
                scan_res = await self.live_automation.force_scan()
                response = f"Scan completed: commands={scan_res.get('commands_processed', 0)} files={scan_res.get('files_ingested', 0)} chunks={scan_res.get('chunks_ingested', 0)}"
                return _respond(response)
            elif lowered.startswith("rag search "):
                query = text[len("rag search "):].strip()
                response = await self.live_automation.search_rag(query)
                return _respond(response)

        if goal_res := await self._handle_goal_intent(lowered, text):
            return _respond(goal_res)

        if pref_res := await self._handle_preference_intent(lowered, text):
            return _respond(pref_res)

        from core.controller.request_rules import is_active_window_request
        if is_active_window_request(lowered):
            obs = await self.desktop_observer.observe()
            title = obs.active_window.get("title", "")
            return _respond(f"The active window is: {title}")

        desktop_plan = plan_desktop_command(text)
        if desktop_plan is not None:
            if not self._app_launch_enabled:
                return _respond(self._app_launch_disabled_message())
            if desktop_res := await handle_desktop_command(text):
                return _respond(desktop_res)

        if (
            self._looks_like_desktop_control_request(lowered)
            and not self._gui_automation_enabled
        ):
            return _respond(self._desktop_control_disabled_message())

        # ── Explicit web-search fast-path ─────────────────────────────────────
        # If the user clearly asks to search / browse the web, skip the LLM
        # planner (which can fail to emit the right tool JSON) and call
        # web_search directly, then synthesise a natural-language reply.
        if self._is_explicit_web_search(lowered):
            self._dashboard_update(state="EXECUTE", last_input=user_input)
            web_response = await self._handle_web_search(text, trace_id=trace_id)
            return _respond(web_response)

        if len(text.split()) > 2 and any(kw in lowered for kw in _AGENTIC_KEYWORDS):
            self._dashboard_update(state="PLANNING", last_input=user_input)
            plan = await self.task_planner.plan(text)
            if plan and plan.get("tools_required"):
                self._dashboard_update(state="EXECUTE", last_input=user_input)
                
                from core.context.context import TaskExecutionContext
                from core.state_machine import StateMachine
                
                event_bus = self.container.resolve("event_bus") if self.container else None
                task_sm = StateMachine(event_bus=event_bus)
                
                def _update_dash_state(old, new):
                    self._dashboard_update(state=new.value)
                task_sm.add_listener(_update_dash_state)
                
                context = TaskExecutionContext(
                    task_id=uuid.uuid4().hex[:8],
                    trace_id=trace_id,
                    state_machine=task_sm,
                )
                
                context_block = await self.memory.build_context_block(text)
                context.set("context_block", context_block)
                
                trace = await self.agent_loop.run(
                    goal=text,
                    context=context,
                )
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
        await self.initialize()
        asyncio.create_task(
            self.monitor.start(),
            name="jarvis_resource_monitor_start",
        )
        if self.live_automation is not None:
            await self.live_automation.start()

    async def run_cli(self) -> None:
        if self._goal_check_task is None or self._goal_check_task.done():
            self._goal_check_task = asyncio.create_task(
                self._check_due_goals(),
                name="jarvis_goal_due_checker",
            )

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
        if self.live_automation is not None:
            await self.live_automation.stop()

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
        self._dashboard_update(state="THINKING", last_input=text)
        response = await self.process(text)
        self._dashboard_update(state="IDLE", last_response=response)
        return response

    async def _check_due_goals(self) -> None:
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
                self._dashboard_update(
                    active_goals=len(self.goal_manager.active_goals())
                )
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
        return extract_goal_delay_seconds(user_input)

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
            self._goals_file.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
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


Controller = JarvisControllerV2

__all__ = ["JarvisControllerV2", "Controller"]
