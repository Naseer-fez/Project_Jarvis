"""
core/agent/controller.py
────────────────────────
MainController — Top-level orchestrator for Jarvis.
Now strictly enforces the StateMachine lifecycle.
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import Optional

# ─── IMPORTS ────────────────────────────────────────────────────────────────

# 1. State Machine (The Brain's Lifecycle)
from core.agent.state_machine import StateMachine, AgentState 
from core.agent.agent_loop import AgentLoopEngine

# 2. Intelligence & Planning
from core.planning.task_planner import TaskPlanner
from core.llm.ollama_llm import OllamaLLM
from core.intents import IntentClassifierV2, Intent

# 3. Tools & Execution
from core.tools.tool_router import ToolRouter
from core.tools.builtin_tools import register_all_tools

# 4. Autonomy & Safety
from core.autonomy.risk_evaluator import RiskEvaluator
from core.autonomy.autonomy_governor import AutonomyGovernor

# 5. Memory & Audit
from core.memory.memory_engine import MemoryEngine
from audit.audit_logger import AuditLogger

# Optional / External Tools
try:
    from core.tools.vision import analyze_image
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

logger = logging.getLogger("Jarvis.Controller")

# Keywords that suggest a complex "Agentic" task
AGENT_TRIGGER_KEYWORDS = {
    "create", "write", "read", "scan", "list", "show", "find", 
    "fetch", "check", "log", "save", "search", "run", "execute", 
    "analyze", "summarize", "delete", "edit", "code", "debug"
}

class MainController:
    def __init__(
        self,
        voice_enabled: bool = False,
        autonomy_level: int = 1,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
    ):
        self.session_id = f"session_{int(time.time())}"
        
        # 1. Initialize Foundation
        self.model = model
        self.voice_enabled = voice_enabled
        self.router = ToolRouter()
        
        # 2. Initialize State Machine
        # This tracks if we are LISTENING, THINKING, ACTING, etc.
        self.sm = StateMachine()
        
        # 3. Initialize Intelligence
        self.llm = OllamaLLM(model=model, base_url=ollama_url)
        self.classifier = IntentClassifierV2(self.llm)
        self.planner = TaskPlanner(model=model, ollama_url=ollama_url)
        self.risk = RiskEvaluator(autonomy_level=autonomy_level)
        self.gov = AutonomyGovernor(level=autonomy_level)
        
        # 4. Initialize Memory & Audit
        self.memory = MemoryEngine(session_id=self.session_id)
        self.audit = AuditLogger(session_id=self.session_id)
        
        # 5. Initialize Agent Loop (Passes SM to it so it can transition to PLANNING/ACTING)
        self.agent_loop = AgentLoopEngine(
            state_machine=self.sm,
            task_planner=self.planner,
            tool_router=self.router,
            risk_evaluator=self.risk,
            autonomy_governor=self.gov,
            model=model,
            ollama_url=ollama_url,
        )

        # 6. Register Tools
        register_all_tools(self.router)
        if HAS_VISION:
            self.router.register("see", analyze_image)
        else:
            logger.warning("Vision tool not loaded (missing dependency or file).")

        # 7. Ensure Environment
        Path("./workspace").mkdir(exist_ok=True)
        Path("./outputs").mkdir(exist_ok=True)
        
        self._running = False

    async def run(self):
        """Main interactive loop."""
        self._running = True
        self._print_banner()

        # Health Check
        if not await self.llm.check_availability():
            print(f"\n⚠️  CRITICAL: Model '{self.model}' not reachable at {self.llm.base_url}")
            print(f"   Run: `ollama run {self.model}` in another terminal.\n")

        # Ensure we start in IDLE
        if self.sm.state != AgentState.IDLE:
            self.sm.force_idle()

        while self._running:
            try:
                # 1. LISTENING Phase
                user_input = await self._get_input()
                if not user_input: continue
                
                # 2. THINKING Phase (Processing)
                await self._handle_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n[Jarvis] Interrupt received. Stopping...")
                break
            except Exception as e:
                logger.exception("Error in main loop")
                print(f"\n❌ Error: {e}")
                self.sm.transition(AgentState.ERROR)
                time.sleep(1)
                self.sm.force_idle()

        await self._shutdown()

    async def _get_input(self) -> str:
        """Transitions IDLE -> LISTENING -> gets input -> returns."""
        # Ensure we are in a valid state to start listening
        if self.sm.state != AgentState.IDLE:
            self.sm.force_idle()
            
        self.sm.transition(AgentState.LISTENING)
        
        loop = asyncio.get_event_loop()
        try:
            # Non-blocking input
            prompt = "\n(Listening) 👤 You: "
            raw = await loop.run_in_executor(None, lambda: input(prompt).strip())
            return raw
        except EOFError:
            self._running = False
            return ""

    async def _handle_input(self, text: str):
        """Transitions LISTENING -> THINKING -> [Action]"""
        # Valid Transition: LISTENING -> THINKING
        self.sm.transition(AgentState.THINKING)
        
        text_lower = text.lower()

        # 1. Fast Path: Hardcoded System Commands
        if text_lower in {"exit", "quit", "bye"}:
            print("\n[Jarvis] Goodbye.")
            self._running = False
            return
            
        if text_lower == "help":
            self._print_help()
            self.sm.transition(AgentState.IDLE) # Done thinking
            return
            
        if text_lower == "status":
            self._print_status()
            self.sm.transition(AgentState.IDLE)
            return

        if text_lower.startswith("autonomy"):
            try:
                level = int(text.split()[-1])
                self.gov.escalate(level)
                self.risk.autonomy_level = level
                print(f"⚙️  Autonomy updated: {self.gov.describe()}")
            except ValueError:
                print("Usage: autonomy <0-3>")
            self.sm.transition(AgentState.IDLE)
            return

        # 2. Intelligent Classification
        print("   (Thinking...)", end="\r")
        classification = self.classifier.classify(text)
        intent = classification.get("intent")
        confidence = classification.get("confidence", 0.0)

        # 3. Routing Logic
        
        # A. System Commands (via LLM classification)
        if intent == Intent.COMMAND.value and confidence > 0.8:
            await self._simple_chat(text, system_override=True)
            return

        # B. Memory Operations
        if intent == Intent.QUERY_MEMORY.value:
            print("   (Searching Memory...)", end="\r")
            mem_hits = self.memory.search(text, limit=3)
            context = "\n".join([m['content'] for m in mem_hits]) if mem_hits else "No specific memory found."
            await self._simple_chat(text, context_inject=f"MEMORY RESULTS:\n{context}")
            return

        if intent == Intent.STORE_MEMORY.value:
            self.memory.store(text, category="user_defined")
            print(f"💾  Memory saved.")
            self.sm.transition(AgentState.IDLE)
            return

        # C. Agent Tasks vs. Chat
        is_task_keyword = any(kw in text_lower for kw in AGENT_TRIGGER_KEYWORDS)
        
        if is_task_keyword and self.gov.level >= 1:
            # THINKING -> PLANNING (Handled inside AgentLoop)
            await self._run_agent_task(text)
        else:
            # THINKING -> SPEAKING
            await self._simple_chat(text)

    async def _simple_chat(self, text: str, context_inject: str = "", system_override: bool = False):
        """Standard LLM response (Chat Mode)."""
        # Ensure we are THINKING (we should be, but safety check)
        if self.sm.state != AgentState.THINKING:
             self.sm.transition(AgentState.THINKING)
        
        # Get context
        if not context_inject:
            context_inject = self.memory.context_summary(text)

        response = await self.llm.chat(text, inject_context=context_inject)
        
        # Transition THINKING -> SPEAKING
        self.sm.transition(AgentState.SPEAKING)
        print(f"🤖 Jarvis: {response}")
        
        # Log
        self.audit.log_voice_interaction("user", text)
        self.audit.log_voice_interaction("jarvis", response)
        self.memory.store(f"User: {text}\nJarvis: {response}", category="conversation")
        
        # Done -> IDLE
        self.sm.transition(AgentState.IDLE)

    async def _run_agent_task(self, goal: str):
        """Execute the full Agentic Loop."""
        # AgentLoop expects to be called while in THINKING or IDLE
        # It will transition to PLANNING -> ... -> ACTING -> ...
        print(f"\n⚙️  Starting Agent Task: {goal!r}")
        
        trace = await self.agent_loop.run(
            goal=goal,
            context=self.memory.context_summary(goal),
            confirm_callback=self._confirm_action
        )

        print(f"\n🤖 Jarvis: {trace.final_response}")
        
        self.audit.log_trace(trace.to_dict())
        if trace.success:
            self.memory.store(f"Completed task: {goal}", category="episodic")
        
        # Force IDLE after complex task to reset state
        self.sm.force_idle()

    async def _confirm_action(self, prompt: str) -> bool:
        """Callback for autonomy governor. Transitions to AWAITING_CONFIRMATION."""
        previous_state = self.sm.state
        self.sm.transition(AgentState.AWAITING_CONFIRMATION)
        
        print(f"\n🛑  PERMISSION REQUIRED: {prompt}")
        choice = await asyncio.get_event_loop().run_in_executor(None, input, "    Allow? [y/N]: ")
        allowed = choice.strip().lower().startswith('y')
        
        # Return to previous state (usually RISK_EVALUATION or PLANNING)
        # Note: The State Machine transitions defines AWAITING -> ACTING or IDLE.
        # We need to be careful here. 
        # Ideally, AgentLoop handles the transition to ACTING if True.
        
        return allowed

    def _print_banner(self):
        print("\n" + "═" * 50)
        print("   J A R V I S   v4.0 (Hybrid Controller)")
        print("═" * 50)
        print(f"   Model    : {self.model}")
        print(f"   Autonomy : {self.gov.describe()}")
        print(f"   Vision   : {'Enabled' if HAS_VISION else 'Disabled'}")
        print("═" * 50 + "\n")

    def _print_help(self):
        print("""
    Commands:
      help          Show this menu
      status        Show system state
      clear         Clear history
      autonomy <N>  Set level (0=Chat, 1=Suggest, 2=ReadOnly, 3=Full)
      exit          Quit
        """)

    def _print_status(self):
        print(f"""
    [System Status]
      State:    {self.sm.state.name}
      Tools:    {len(self.router.registered_tools())} registered
      Memory:   {len(self.memory._entries)} items (Session)
      Log File: {self.audit.session_file}
        """)

    async def _shutdown(self):
        print("\n[Jarvis] Saving session artifacts...")
        self.audit.log_memory_snapshot(self.memory.snapshot())
        print("[Jarvis] Shut down complete.")