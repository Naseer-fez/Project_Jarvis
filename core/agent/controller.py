"""
MainController — top-level orchestrator for Jarvis.
Wires all components together and runs the interactive input loop.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from core.state_machine import StateMachine, AgentState
from core.task_planner import TaskPlanner
from core.tool_router import ToolRouter
from core.risk_evaluator import RiskEvaluator
from core.autonomy_governor import AutonomyGovernor
from core.agent_loop import AgentLoopEngine
from core.ollama_llm import OllamaLLM
from core.audit_logger import AuditLogger
from memory.memory_engine import MemoryEngine
from tools.builtin_tools import register_all_tools

logger = logging.getLogger("Jarvis.Controller")

STOP_KEYWORDS = {"stop", "cancel", "abort", "quit", "exit", "bye"}

# Intent signals that trigger the full agent loop vs. simple chat
AGENT_TRIGGER_KEYWORDS = [
    "create", "write", "read", "list", "show", "find", "get", "fetch",
    "check", "log", "save", "search", "run", "execute", "do", "plan",
    "analyze", "summarize", "open", "delete",
]


def _is_agent_task(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in AGENT_TRIGGER_KEYWORDS)


class MainController:
    def __init__(
        self,
        voice_enabled: bool = False,
        autonomy_level: int = 1,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
    ):
        session_id = f"session_{int(time.time())}"

        # Core components
        self.sm = StateMachine()
        self.llm = OllamaLLM(model=model, base_url=ollama_url)
        self.planner = TaskPlanner(model=model, ollama_url=ollama_url)
        self.router = ToolRouter()
        self.risk = RiskEvaluator(autonomy_level=autonomy_level)
        self.gov = AutonomyGovernor(level=autonomy_level)
        self.memory = MemoryEngine(session_id=session_id)
        self.audit = AuditLogger(session_id=session_id)
        self.agent_loop = AgentLoopEngine(
            state_machine=self.sm,
            task_planner=self.planner,
            tool_router=self.router,
            risk_evaluator=self.risk,
            autonomy_governor=self.gov,
            model=model,
            ollama_url=ollama_url,
        )

        register_all_tools(self.router)

        self.voice_enabled = voice_enabled
        self.model = model
        self._running = False

        # Create required directories
        Path("./workspace").mkdir(exist_ok=True)
        Path("./outputs").mkdir(exist_ok=True)

    async def run(self):
        """Main run loop."""
        self._running = True

        # Startup banner
        print("\n" + "═" * 60)
        print("  J A R V I S  —  Local Agentic Assistant")
        print("═" * 60)
        print(f"  Model   : {self.model} (via Ollama)")
        print(f"  Autonomy: {self.gov.describe()}")
        print(f"  Voice   : {'enabled' if self.voice_enabled else 'disabled'}")
        print("  Type 'help' for commands | Ctrl+C to exit")
        print("═" * 60 + "\n")

        # Check Ollama
        available = await self.llm.check_availability()
        if not available:
            print(f"⚠️  Warning: Ollama model '{self.model}' may not be available.")
            print(f"   Run: ollama pull {self.model}\n")

        # Voice daemon (stub — wired for future voice module)
        if self.voice_enabled:
            print("🎤 Voice mode enabled (STT/TTS stubs active)\n")

        while self._running:
            try:
                user_input = await self._get_input()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            await self._handle_input(user_input)

        await self._shutdown()

    async def _get_input(self) -> str:
        """Get input from text (or voice in future)."""
        loop = asyncio.get_event_loop()
        self.sm.transition(AgentState.IDLE) if self.sm.state != AgentState.IDLE else None
        raw = await loop.run_in_executor(None, lambda: input("You: ").strip())
        self.audit.log_voice_interaction("user", raw)
        return raw

    async def _handle_input(self, text: str):
        """Route input to the right handler."""
        lower = text.lower().strip()

        # Built-in commands
        if lower in STOP_KEYWORDS:
            if self.agent_loop and self.sm.is_interruptible():
                self.agent_loop.request_interrupt()
                print("\n[Jarvis] Stopping current task...\n")
            else:
                print("\n[Jarvis] Goodbye.\n")
                self._running = False
            return

        if lower == "help":
            self._print_help()
            return

        if lower == "status":
            self._print_status()
            return

        if lower.startswith("autonomy "):
            level_str = lower.split()[-1]
            if level_str.isdigit():
                self.gov.escalate(int(level_str))
                self.risk.autonomy_level = self.gov.level
                print(f"\n[Jarvis] {self.gov.describe()}\n")
            return

        if lower == "memory":
            for e in self.memory.recent(5):
                print(f"  [{e.category}] {e.content}")
            print()
            return

        if lower == "clear":
            self.llm.reset_history()
            print("\n[Jarvis] Conversation history cleared.\n")
            return

        # Route to agent loop or simple chat
        if _is_agent_task(text) and self.gov.level >= 1:
            await self._run_agent_task(text)
        else:
            await self._simple_chat(text)

    async def _simple_chat(self, text: str):
        """Pure conversational response via LLM."""
        self.sm.transition(AgentState.THINKING)
        context = self.memory.context_summary(text)
        response = await self.llm.chat(text, inject_context=context)
        self.sm.transition(AgentState.SPEAKING)
        print(f"\nJarvis: {response}\n")
        self.audit.log_voice_interaction("jarvis", response)
        self.memory.store(f"User asked: {text[:100]}", category="episodic")
        self.sm.transition(AgentState.IDLE)

    async def _run_agent_task(self, goal: str):
        """Run the full agent loop for an action-oriented goal."""
        print(f"\n[Jarvis] Planning: {goal!r}\n")
        context = self.memory.context_summary(goal)

        trace = await self.agent_loop.run(
            goal=goal,
            context=context,
            confirm_callback=self._confirm_callback,
        )

        print(f"\nJarvis: {trace.final_response}\n")
        self.audit.log_voice_interaction("jarvis", trace.final_response)
        self.audit.log_trace(trace.to_dict())

        if trace.plan:
            self.audit.log_plan(trace.plan)

        for obs in self.router.get_observations():
            self.audit.log_observation(obs.to_dict())

        for risk in trace.risk_scores:
            self.audit.log_risk(risk)

        if trace.reflection:
            self.audit.log_reflection(trace.reflection)
            self.memory.store(
                f"Completed task: {goal[:80]} — {trace.stop_reason}",
                category="episodic",
                tags=["task", "completed"],
            )

        self.audit.log_memory_snapshot(self.memory.snapshot())

    async def _confirm_callback(self, prompt: str) -> bool:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, input, prompt)
        return answer.strip().lower() in ("y", "yes")

    def _print_help(self):
        print("""
╔══════════════════════════════════════════════════╗
║              Jarvis — Commands                   ║
╠══════════════════════════════════════════════════╣
║ help           Show this help menu               ║
║ status         Show current state & config       ║
║ memory         Show recent memory entries        ║
║ clear          Clear conversation history        ║
║ autonomy <N>   Set autonomy level (0-3)          ║
║ stop/cancel    Interrupt current task            ║
║ exit/quit      End session                       ║
╚══════════════════════════════════════════════════╝
Autonomy levels:
  0 = Chat only     (no tools)
  1 = Suggest only  (describes actions, doesn't run them)
  2 = Read-only     (can query files/system/memory)
  3 = Write+confirm (can write files after your approval)
""")

    def _print_status(self):
        print(f"""
[Status]
  State    : {self.sm.state.name}
  Model    : {self.model}
  Autonomy : {self.gov.describe()}
  Tools    : {', '.join(self.router.registered_tools())}
  Memory   : {len(self.memory._entries)} entries
""")

    async def _shutdown(self):
        logger.info("Shutting down Jarvis...")
        self.audit.log_memory_snapshot(self.memory.snapshot())
        print("\n[Jarvis] Session ended. Artifacts saved to ./outputs/Jarvis-Session/\n")

