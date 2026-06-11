"""JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes."""

from __future__ import annotations

import asyncio
import configparser
import logging
import uuid
from typing import Any

from core.base_controller import BaseController
from core.controller.intents import (
    handle_goal_intent,
    handle_preference_intent,
)
from core.controller.intent_router import IntentRouter
from core.controller.services import build_controller_services
from core.llm.defaults import DEFAULT_MODEL

# Import extracted facade components
from core.controller.llm_dispatcher import LLMDispatcher
from core.controller.goal_runner import GoalRunner

# Import extracted subsystems
from core.controller.llm_orchestrator import LLMOrchestrator
from core.controller.memory_subsystem import MemorySubsystem
from core.controller.automation_manager import AutomationManager

logger = logging.getLogger(__name__)

class JarvisControllerV2(BaseController):
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
        super().__init__()
        
        self.config = (
            config
            if isinstance(config, configparser.ConfigParser)
            else configparser.ConfigParser()
        )
        for section in ["execution", "memory", "automation", "voice"]:
            if not self.config.has_section(section):
                self.config.add_section(section)
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

        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._on_state_update = lambda **_: None
        self.exchanges = 0
        self._state_lock = asyncio.Lock()
        self.voice_loop = None

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

        # Initialize facades
        self.llm_dispatcher = LLMDispatcher(
            llm=self.llm,
            model_router=self.model_router,
            memory=self.memory,
            profile=self.profile
        )
        
        self.goal_runner = GoalRunner(
            goal_manager=self.goal_manager,
            scheduler=self.scheduler,
            notifier=self.notifier,
            voice_layer=self._voice_layer,
            goals_file=settings.goals_file,
            goal_check_interval_seconds=settings.goal_check_interval_seconds,
            dashboard_update_cb=self._dashboard_update
        )
        self.goal_runner.load_goal_state()
        self._goal_check_task: asyncio.Task | None = None

        # Initialize Subsystems
        self.llm_orchestrator = LLMOrchestrator(self.llm_dispatcher)
        self.memory_subsystem = MemorySubsystem(self.memory, self.profile, self.synthesizer, self.config)
        
        async def cmd_handler(cmd: str) -> str:
            return await self.process(cmd)
            
        self.automation_manager = AutomationManager(
            config=self.config,
            memory=self.memory,
            llm=self.llm,
            notifier=self.notifier,
            desktop_observer=self.desktop_observer,
            container=self.container,
            command_handler=cmd_handler
        )

        self.register_subsystem(self.llm_orchestrator)
        self.register_subsystem(self.memory_subsystem)
        self.register_subsystem(self.automation_manager)

        self.intent_router = IntentRouter()
        self._setup_intent_routes()

    async def initialize(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "memory_mode": getattr(self.memory_subsystem, "memory_status", {}).get("mode"),
            "memory_init": getattr(self.memory_subsystem, "memory_status", {}),
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
            self.goal_runner.persist_goal_state()
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
        classification = getattr(self, "current_classification", {})
        return await self.llm_orchestrator.dispatch(text, classification, self.session_id, trace_id)

    def _looks_like_desktop_control_request(self, lowered: str) -> bool:
        return any(k in lowered for k in ["click", "desktop", "mouse", "keyboard", "screen", "type"])

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
        from core.controller.intent_handlers import register_intent_routes
        register_intent_routes(self)

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
            async with self._state_lock:
                self.memory_subsystem.update_profile(user_input, response)

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

    async def startup(self) -> None:
        self._runtime_loop = asyncio.get_running_loop()
        await super().startup()
        await self.monitor.start()
        if self._goal_check_task is None or self._goal_check_task.done():
            self._goal_check_task = asyncio.create_task(
                self.goal_runner.check_due_goals(),
                name="jarvis_goal_due_checker",
            )

    async def start(self) -> None:
        """Alias for startup() to maintain backward compatibility."""
        await self.startup()

    async def run_cli(self) -> None:
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

        await super().shutdown()

    def _dashboard_update(self, **kwargs: Any) -> None:
        try:
            self._on_state_update(**kwargs)
        except Exception as e:
            logger.warning("Failed to update dashboard state: %s", e, exc_info=True)

Controller = JarvisControllerV2

__all__ = ["JarvisControllerV2", "Controller"]
