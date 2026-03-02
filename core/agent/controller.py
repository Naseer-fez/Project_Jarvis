"""Top-level agent controller wiring planner, risk, tools, and integrations."""

from __future__ import annotations

import asyncio
import configparser
import logging
import time
from pathlib import Path
from typing import Optional

from core.agent.agent_loop import AgentLoopEngine
from core.agent.state_machine import AgentState, StateMachine
from core.autonomy.autonomy_governor import AutonomyGovernor
from core.autonomy.risk_evaluator import RiskEvaluator
from core.llm.ollama_llm import OllamaLLM
from core.llm.task_planner import TaskPlanner
from core.memory.memory_engine import MemoryEngine
from core.planning.intents import Intent, IntentClassifierV2
from core.tools.builtin_tools import register_all_tools
from core.tools.tool_router import ToolRouter
from integrations.loader import load_all
from integrations.registry import integration_registry

try:
    from audit.audit_logger import AuditLogger
except Exception:  # noqa: BLE001
    from core.logging.audit_logger import AuditLogger

logger = logging.getLogger("Jarvis.Controller")

try:
    from core.tools.vision import analyze_image

    HAS_VISION = True
except Exception:  # noqa: BLE001
    HAS_VISION = False

AGENT_TRIGGER_KEYWORDS = {
    "create",
    "write",
    "read",
    "scan",
    "list",
    "show",
    "find",
    "fetch",
    "check",
    "log",
    "save",
    "search",
    "run",
    "execute",
    "analyze",
    "summarize",
    "delete",
    "edit",
    "code",
    "debug",
}


class MainController:
    def __init__(
        self,
        config: Optional[configparser.ConfigParser] = None,
        voice_enabled: bool = False,
        autonomy_level: int = 1,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
    ):
        self.session_id = f"session_{int(time.time())}"
        self.config = config or configparser.ConfigParser()

        self.model = model
        self.voice_enabled = voice_enabled

        self.sm = StateMachine()
        self.router = ToolRouter()

        self.llm = OllamaLLM(model=model, base_url=ollama_url)
        self.classifier = IntentClassifierV2(self.llm)
        self.planner = TaskPlanner(self.config)
        self.risk = RiskEvaluator(config=self.config if self.config.sections() else None)
        self.gov = AutonomyGovernor(level=autonomy_level)

        self.memory = MemoryEngine(session_id=self.session_id)
        self.audit = AuditLogger(session_id=self.session_id)

        max_iterations = int(self.config.get("agent", "max_iterations", fallback="10"))
        self.agent_loop = AgentLoopEngine(
            state_machine=self.sm,
            task_planner=self.planner,
            tool_router=self.router,
            risk_evaluator=self.risk,
            autonomy_governor=self.gov,
            model=model,
            ollama_url=ollama_url,
            max_iterations=max_iterations,
        )

        register_all_tools(self.router)
        if HAS_VISION:
            self.router.register("see", analyze_image)

        self.integration_registry = integration_registry
        try:
            summary = load_all(self.config, self.integration_registry)
            logger.info("Integration load summary: %s", summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Integration loader failed: %s", exc)

        Path("./workspace").mkdir(exist_ok=True)
        Path("./outputs").mkdir(exist_ok=True)

        self._running = False

    async def run(self) -> None:
        self._running = True
        self._print_banner()

        if not await self.llm.check_availability():
            print(
                f"\nWarning: model '{self.model}' is not reachable at {self.llm.base_url}. "
                f"Run: ollama run {self.model}\n"
            )

        if self.sm.state != AgentState.IDLE:
            self.sm.force_idle()

        while self._running:
            try:
                user_input = await self._get_input()
                if not user_input:
                    continue
                await self._handle_input(user_input)
            except KeyboardInterrupt:
                print("\n[Jarvis] Interrupt received. Stopping...")
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Controller loop error")
                print(f"Error: {exc}")
                self.sm.transition(AgentState.ERROR)
                self.sm.force_idle()

        await self._shutdown()

    async def _get_input(self) -> str:
        if self.sm.state != AgentState.IDLE:
            self.sm.force_idle()

        self.sm.transition(AgentState.LISTENING)
        loop = asyncio.get_running_loop()

        try:
            return await loop.run_in_executor(None, lambda: input("\nYou: ").strip())
        except EOFError:
            self._running = False
            return ""

    async def _handle_input(self, text: str) -> None:
        self.sm.transition(AgentState.THINKING)
        lowered = text.lower()

        if lowered in {"exit", "quit", "bye"}:
            print("\n[Jarvis] Goodbye.")
            self._running = False
            return

        if lowered == "help":
            self._print_help()
            self.sm.transition(AgentState.IDLE)
            return

        if lowered == "status":
            self._print_status()
            self.sm.transition(AgentState.IDLE)
            return

        if lowered.startswith("autonomy"):
            try:
                level = int(text.split()[-1])
                self.gov.escalate(level)
                print(f"Autonomy updated: {self.gov.describe()}")
            except Exception:  # noqa: BLE001
                print("Usage: autonomy <0-3>")
            self.sm.transition(AgentState.IDLE)
            return

        classification = self.classifier.classify(text)
        intent = classification.get("intent")
        confidence = float(classification.get("confidence", 0.0) or 0.0)

        if intent == Intent.COMMAND.value and confidence > 0.8:
            await self._simple_chat(text)
            return

        if intent == Intent.QUERY_MEMORY.value:
            hits = self.memory.search(text, limit=3)
            context = "\n".join(item["content"] for item in hits) if hits else "No memory matches found."
            await self._simple_chat(text, context_inject=f"MEMORY RESULTS:\n{context}")
            return

        if intent == Intent.STORE_MEMORY.value:
            self.memory.store(text, category="user_defined")
            print("Memory saved.")
            self.sm.transition(AgentState.IDLE)
            return

        is_task = any(keyword in lowered for keyword in AGENT_TRIGGER_KEYWORDS)
        if is_task and self.gov.level >= 1:
            await self._run_agent_task(text)
        else:
            await self._simple_chat(text)

    async def _simple_chat(self, text: str, context_inject: str = "") -> None:
        if self.sm.state != AgentState.THINKING:
            self.sm.transition(AgentState.THINKING)

        if not context_inject:
            context_inject = self.memory.context_summary(text)

        response = await self.llm.chat(text, inject_context=context_inject)

        self.sm.transition(AgentState.SPEAKING)
        print(f"Jarvis: {response}")

        self.audit.log_voice_interaction("user", text)
        self.audit.log_voice_interaction("jarvis", response)
        self.memory.store(f"User: {text}\nJarvis: {response}", category="conversation")

        self.sm.transition(AgentState.IDLE)

    async def _run_agent_task(self, goal: str) -> None:
        print(f"\nStarting agent task: {goal!r}")

        trace = await self.agent_loop.run(
            goal=goal,
            context=self.memory.context_summary(goal),
            confirm_callback=self._confirm_action,
        )

        print(f"\nJarvis: {trace.final_response}")
        self.audit.log_trace(trace.to_dict())

        if trace.success:
            self.memory.store(f"Completed task: {goal}", category="episodic")

        self.sm.force_idle()

    async def _confirm_action(self, prompt: str) -> bool:
        self.sm.transition(AgentState.AWAITING_CONFIRMATION)
        print(f"\nPermission required: {prompt}")
        answer = await asyncio.get_running_loop().run_in_executor(None, input, "Allow? [y/N]: ")
        return str(answer).strip().lower().startswith("y")

    def _print_banner(self) -> None:
        print("\n" + "=" * 50)
        print("   J A R V I S   v4.0")
        print("=" * 50)
        print(f"   Model    : {self.model}")
        print(f"   Autonomy : {self.gov.describe()}")
        print(f"   Vision   : {'Enabled' if HAS_VISION else 'Disabled'}")
        print("=" * 50 + "\n")

    def _print_help(self) -> None:
        print(
            """
Commands:
  help          Show this menu
  status        Show system state
  autonomy <N>  Set level (0=Chat, 1=Suggest, 2=ReadOnly, 3=Full)
  exit          Quit
            """.strip()
        )

    def _print_status(self) -> None:
        print(
            f"""
[System Status]
  State:    {self.sm.state.name}
  Tools:    {len(self.router.registered_tools())} registered
  Memory:   {len(self.memory._entries)} items (session)
  Log File: {self.audit.session_file}
            """.strip()
        )

    async def _shutdown(self) -> None:
        print("\n[Jarvis] Saving session artifacts...")
        self.audit.log_memory_snapshot(self.memory.snapshot())
        print("[Jarvis] Shut down complete.")


__all__ = ["MainController"]
