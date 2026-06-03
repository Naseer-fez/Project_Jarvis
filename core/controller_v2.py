"""JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes."""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from core.controller.intents import (
    handle_goal_intent,
    handle_preference_intent,
)
from core.controller.intent_router import IntentRouter
from core.controller.services import build_controller_services
from core.desktop.shortcuts import handle_desktop_command, plan_desktop_command
from core.llm.defaults import DEFAULT_MODEL

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
        services: Any = None,
        settings: Any = None,
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

        self._inflight_llm_calls = 0

        if services is None or settings is None:
            settings, services = build_controller_services(
                self.config,
                container=container,
                db_path=db_path,
                chroma_path=chroma_path,
                model_name=model_name,
                embedding_model=embedding_model,
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
        self._state_lock = asyncio.Lock()
        self.voice_loop = None
        self._goal_check_task: asyncio.Task | None = None
        self._goal_check_interval_seconds = settings.goal_check_interval_seconds
        self._goals_file = settings.goals_file
        self._load_goal_state()
        self._synthesis_tasks: set[asyncio.Task] = set()

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
                    dag_executor=self.container.resolve("dag_executor") if self.container and self.container.has("dag_executor") else None,
                )
            except Exception as exc:
                logger.warning("Failed to initialize LiveAutomationEngine: %s", exc, exc_info=True)

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
                logger.warning("Voice unavailable: %s", exc, exc_info=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VoiceLayer init failed: %s", exc, exc_info=True)

        self.intent_router = IntentRouter()
        self._setup_intent_routes()

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
        self._inflight_llm_calls += 1
        classification = getattr(self, "current_classification", {})
        complexity = classification.get("complexity", 0.5)
        
        try:
            profile_summary = ""
            
            # Selective context injection
            if complexity > 0.2:
                await self.memory.build_context_block(text)
                profile_summary = (
                    self.profile.get_communication_style() if self.profile else ""
                )

            # Map class route to task_type
            task_type = classification.get("route", "chat")
            if task_type == "direct":
                task_type = "reflex"
            elif task_type == "premium":
                task_type = "deep_reasoning"
            elif task_type == "planner":
                task_type = "planning"

            selected_model = self.model_router.get_best_available(task_type)
            self.llm.model = selected_model

            messages = [{"role": "user", "content": text}]

            logger.info("Dispatching to LLM: %r", selected_model, extra={"trace_id": trace_id})
            response = await self.llm.chat_async(
                messages,
                query_for_memory=text if complexity > 0.2 else "",
                profile_summary=profile_summary,
                trace_id=trace_id,
            )

            if not response or response == "LLM Offline.":
                from core.controller.request_rules import is_preference_relevant
                prefs = await self.memory.recall_preferences(text, top_k=5)
                for pref in prefs:
                    val = pref.get("value")
                    key = pref.get("key")
                    if val and key and is_preference_relevant(key, text):
                        return f"Offline fallback from memory: {val}"
                return "I don't know while offline."

            await self.memory.store_conversation(text, response, self.session_id)
            return response
        finally:
            self._inflight_llm_calls -= 1

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


    def _setup_intent_routes(self) -> None:
        async def handle_status(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            return f"Session: {ctx.session_id} | Memory Mode: {ctx.memory.mode}"
        self.intent_router.register(lambda _l, u, c: _l == "status", handle_status)

        async def handle_help(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            return "Commands: status, help, exit, remember <fact>, what's <query>, open <app>, search <query> in <browser>"
        self.intent_router.register(lambda _l, u, c: _l == "help", handle_help)

        async def handle_automation(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            if not getattr(ctx, "live_automation", None):
                return None
            if lowered == "automation status":
                status_info = ctx.live_automation.status()
                return f"{ctx.live_automation.status_line()}\nDrop Root: {status_info.get('drop_root')}\nCommands Dir: {status_info.get('commands_dir')}\nRAG Dir: {status_info.get('rag_dir')}"
            elif lowered == "automation scan":
                scan_res = await ctx.live_automation.force_scan()
                return f"Scan completed: commands={scan_res.get('commands_processed', 0)} files={scan_res.get('files_ingested', 0)} chunks={scan_res.get('chunks_ingested', 0)}"
            elif lowered.startswith("rag search "):
                query = user_input[len("rag search "):].strip()
                return await ctx.live_automation.search_rag(query)
            return None
        self.intent_router.register(lambda _l, u, c: getattr(c, "live_automation", None) is not None and (_l in ("automation status", "automation scan") or _l.startswith("rag search ")), handle_automation)

        async def handle_goal(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            return await ctx._handle_goal_intent(lowered, user_input)
        # Always run, returns None if not matched inside
        self.intent_router.register(lambda _l, u, c: True, handle_goal)

        async def handle_pref(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            return await ctx._handle_preference_intent(lowered, user_input)
        self.intent_router.register(lambda _l, u, c: True, handle_pref)

        async def handle_active_window(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            from core.controller.request_rules import is_active_window_request
            if is_active_window_request(lowered):
                obs = await ctx.desktop_observer.observe()
                title = obs.active_window.get("title", "")
                return f"The active window is: {title}"
            return None
        self.intent_router.register(lambda _l, u, c: True, handle_active_window)

        async def handle_desktop_plan(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            desktop_plan = plan_desktop_command(user_input)
            if desktop_plan is not None:
                if not ctx._app_launch_enabled:
                    return ctx._app_launch_disabled_message()
                return await handle_desktop_command(user_input)
            return None
        self.intent_router.register(lambda _l, u, c: True, handle_desktop_plan)

        async def handle_desktop_disabled(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            if ctx._looks_like_desktop_control_request(lowered) and not ctx._gui_automation_enabled:
                return ctx._desktop_control_disabled_message()
            return None
        self.intent_router.register(lambda _l, u, c: True, handle_desktop_disabled)

        async def handle_explicit_web(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            from core.controller.request_rules import is_explicit_web_search
            if is_explicit_web_search(lowered):
                ctx._dashboard_update(state="EXECUTE", last_input=user_input)
                try:
                    from core.controller.web_search import handle_web_search
                except ImportError as exc:
                    logger.error("Failed to import handle_web_search from core.controller.web_search: %s", exc, exc_info=True)
                    return await ctx._dispatch_llm(user_input, trace_id=uuid.uuid4().hex[:8])

                web_response = await handle_web_search(
                    user_input=user_input, 
                    trace_id=uuid.uuid4().hex[:8], 
                    memory=ctx.memory, 
                    llm=ctx.llm, 
                    model_router=ctx.model_router, 
                    profile=ctx.profile
                )
                if web_response:
                    await ctx.memory.store_conversation(user_input, web_response, ctx.session_id)
                    return web_response
            return None
        self.intent_router.register(lambda _l, u, c: True, handle_explicit_web)

        async def handle_agentic(lowered: str, user_input: str, ctx: JarvisControllerV2) -> str | None:
            classification = getattr(ctx, "current_classification", {})
            if not classification.get("skip_planner", False) and classification.get("route") in ("planner", "premium", "full stack"):
                ctx._dashboard_update(state="PLANNING", last_input=user_input)
                
                # Let task_planner also know about the complexity
                plan = await ctx.task_planner.plan(user_input)
                if plan and plan.get("tools_required"):
                    ctx._dashboard_update(state="EXECUTE", last_input=user_input)
                    
                    task_sm = ctx.container.resolve("state_machine") if ctx.container else None
                    if not task_sm:
                        raise RuntimeError("state_machine not found in container")
                    
                    def _update_dash_state(old, new):
                        ctx._dashboard_update(state=new.value)
                    task_sm.add_listener(_update_dash_state)
                    
                    try:
                        context = ctx.container.resolve(
                            "task_execution_context",
                            task_id=uuid.uuid4().hex[:8],
                            trace_id=uuid.uuid4().hex[:8],
                            state_machine=task_sm,
                        )
                        
                        # Only build massive context if it's high complexity
                        if classification.get("complexity", 0.5) > 0.5:
                            context_block = await ctx.memory.build_context_block(user_input)
                            context.set("context_block", context_block)
                        else:
                            context.set("context_block", "")
                        
                        trace = await ctx.agent_loop.run(
                            goal=user_input,
                            context=context,
                        )
                        return trace.final_response
                    finally:
                        if hasattr(task_sm, "remove_listener"):
                            task_sm.remove_listener(_update_dash_state)
            return None
        self.intent_router.register(lambda _l, u, c: True, handle_agentic)


    async def process(self, user_input: str, trace_id: str | None = None) -> str:
        if not trace_id:
            trace_id = uuid.uuid4().hex[:8]
        logger.info("Controller process started", extra={"trace_id": trace_id})

        self._dashboard_update(state="THINKING", last_input=user_input)
        
        async with self._state_lock:
            self.exchanges += 1
            text = (user_input or "").strip()
            if len(text) > 4000:
                text = text[:4000]
            
        lowered = text.lower()
        
        try:
            from core.controller.complexity_scorer import classify_request
            self.current_classification = classify_request(text)
        except ImportError:
            self.current_classification = {"class": "Chat", "complexity": 0.4, "skip_planner": False, "route": "chat"}

        async def _respond(response: str) -> str:
            try:
                async with self._state_lock:
                    self.profile.update_from_conversation(user_input, response)
                    self._conversation_buffer.append(
                        f"User: {user_input}\nJarvis: {response}"
                    )

                    if self.synthesizer.should_run(self.profile):
                        self._schedule_synthesis(self._conversation_buffer[-20:])
                        self._conversation_buffer.clear()
                    elif len(self._conversation_buffer) > 50:
                        self._conversation_buffer = self._conversation_buffer[-50:]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Profile update/synthesis scheduling failed: %s",
                    exc,
                    exc_info=True
                )

            self._dashboard_update(
                state="IDLE",
                last_response=response,
                active_goals=len(self.goal_manager.active_goals()),
            )
            return response

        routed_response = await self.intent_router.route(lowered, text, self)
        if routed_response is not None:
            return await _respond(routed_response)

        response = await self._dispatch_llm(text, trace_id=trace_id)
        return await _respond(response)

    def session_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exchanges": self.exchanges,
        }

    async def start(self) -> None:
        self._runtime_loop = asyncio.get_running_loop()
        await self.initialize()
        await self.monitor.start()
        if self._goal_check_task is None or self._goal_check_task.done():
            self._goal_check_task = asyncio.create_task(
                self._check_due_goals(),
                name="jarvis_goal_due_checker",
            )
        if self.live_automation is not None:
            await self.live_automation.start()

    async def run_cli(self) -> None:
        # Note: _goal_check_task is already started in start()

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

        for task in list(self._synthesis_tasks):
            task.cancel()
        if self._synthesis_tasks:
            try:
                await asyncio.gather(*self._synthesis_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning("Error gathering synthesis tasks during shutdown: %s", e, exc_info=True)

        while getattr(self, "_inflight_llm_calls", 0) > 0:
            await asyncio.sleep(0.1)

        if hasattr(self, "memory") and self.memory is not None:
            try:
                await self.memory.close()
            except Exception:
                logger.warning("Memory cleanup error during shutdown", exc_info=True)

    async def _check_due_goals(self) -> None:
        backoff = 1.0
        while True:
            try:
                await asyncio.sleep(self._goal_check_interval_seconds)
                due_items = self.scheduler.due()
                for item in due_items:
                    msg = f"Due: {item.description or item.goal_id}"
                    self.notifier.notify(msg)
                    item.mark_completed()
                    if self._voice_layer is not None:
                        try:
                            await self._speak_via_voice_layer(msg)
                        except Exception as e:
                            logger.warning("Failed to speak due goal via voice layer: %s", e, exc_info=True)
                if due_items:
                    self._persist_goal_state()
                self._dashboard_update(
                    active_goals=len(self.goal_manager.active_goals())
                )
                backoff = 1.0  # reset backoff
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Goal check loop error: %s", e, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

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
        
        def _track(task):
            self._synthesis_tasks.add(task)
            task.add_done_callback(self._synthesis_tasks.discard)

        try:
            task = asyncio.create_task(coro)
            _track(task)
            return
        except RuntimeError:
            pass

        if self._runtime_loop is not None and self._runtime_loop.is_running():
            def _create_and_track():
                t = asyncio.create_task(coro)
                _track(t)
            self._runtime_loop.call_soon_threadsafe(_create_and_track)
            return

        coro.close()
        logger.warning("No running loop available; skipped profile synthesis task.")



    def _dashboard_update(self, **kwargs: Any) -> None:
        try:
            self._on_state_update(**kwargs)
        except Exception as e:
            logger.warning("Failed to update dashboard state: %s", e, exc_info=True)

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
            logger.warning("Failed to persist goals: %s", exc, exc_info=True)

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
            logger.warning("Failed to load goals: %s", exc, exc_info=True)


Controller = JarvisControllerV2

__all__ = ["JarvisControllerV2", "Controller"]
