"""
core/controller.py — THE SPINE.

Async event loop that orchestrates all V1 components and hosts the V2 voice loop.

Responsibilities:
  - Owns the FSM, memory, planner, risk evaluator, vision tool
  - Provides plan_for_intent() used by both CLI and voice loop
  - Starts the voice loop if --voice flag is passed
  - Runs the CLI input loop
  - Manages graceful shutdown

The voice loop is an OPTIONAL addon — removing it leaves a fully functional V1 system.
"""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import core.logger as logger_mod
from core.state_machine import StateMachine, State, IllegalTransitionError
from core.risk_evaluator import RiskEvaluator
from core.memory.hybrid_memory import HybridMemory
from core.planning.task_planner import TaskPlanner
from core.tools.vision import VisionTool

log = logging.getLogger("jarvis.controller")

class Controller:
    def __init__(self, config: configparser.ConfigParser, voice: bool = False) -> None:
        self._config    = config
        self._voice_on  = voice

        # ── V1 Core ───────────────────────────────────────────────────────────
        self.fsm         = StateMachine()
        self.memory      = HybridMemory(config)
        self.planner     = TaskPlanner(config)
        self.risk        = RiskEvaluator(config)
        self.vision      = VisionTool(config)

        # ── V2 Voice (optional) ───────────────────────────────────────────────
        self._voice_loop: Optional[Any] = None

        # Listen for state transitions and log them (if FSM supports listeners)
        if hasattr(self.fsm, "add_listener"):
            self.fsm.add_listener(self._on_state_change)

        self._running = False

    # ── Audit helper ──────────────────────────────────────────────────────────

    def audit(self, event_type: str, payload: dict) -> str:
        return logger_mod.audit(event_type, payload)

    # ── State listener ────────────────────────────────────────────────────────

    def _on_state_change(self, old: State, new: State) -> None:
        log.debug(f"State: {old.name} → {new.name}")
        self.audit("STATE_TRANSITION", {"from": old.name, "to": new.name})

    # ── Core planner (used by CLI and voice loop) ─────────────────────────────

    def plan_for_intent(self, intent: str) -> Tuple[dict, Any]:
        """
        Generate a plan for intent and evaluate its risk.
        Returns (plan_dict, RiskResult).
        Blocking — run in executor from async context.
        """
        # Build context from hybrid memory (Semantic + SQLite)
        try:
            # Uses HybridMemory's recall method from Session 4
            context_text = self.memory.recall(intent)
        except Exception as e:
            log.warning(f"Memory recall failed during planning: {e}")
            context_text = ""

        context = f"Known context:\n{context_text}" if context_text else ""

        # Generate plan
        plan = self.planner.plan(intent, context=context)
        
        # Evaluate risk based on planned actions
        actions = [step.get("action") for step in plan.get("steps", [])]
        risk = self.risk.evaluate(actions)

        self.audit("PLAN_GENERATED", {
            "intent":        intent,
            "summary":       plan.get("summary", ""),
            "steps":         plan.get("steps", []),
            "risk_level":    risk.level.name,
            "risk_blocked":  risk.is_blocked,
            "risk_reasons":  risk.reasons() if hasattr(risk, 'reasons') else []
        })

        return plan, risk

    # ── Startup / shutdown ────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self.audit("JARVIS_START", {
            "version": self._config.get("general", "version", fallback="2.0.0"),
            "voice":   self._voice_on,
        })

        # Verify Ollama
        try:
            if self.planner.ping():
                log.info("✅ Ollama reachable")
            else:
                log.warning("⚠  Ollama not reachable — planner will degrade gracefully")
        except AttributeError:
            log.debug("Planner has no ping method, skipping Ollama check.")

        # Start voice loop
        if self._voice_on:
            try:
                from core.voice.voice_loop import VoiceLoop
                self._voice_loop = VoiceLoop(self, self._config)
                self._voice_loop.start()
                log.info("✅ Voice loop active")
            except Exception as exc:
                log.error(f"Voice loop failed to start: {exc} — falling back to CLI only")
                self._voice_on = False

        self._print_banner()

    async def shutdown(self) -> None:
        self._running = False
        log.info("Shutting down...")

        if self._voice_loop:
            await self._voice_loop.stop()

        self.audit("JARVIS_SHUTDOWN", {"ts": time.time()})
        
        # Transition to IDLE/Shutdown state gracefully
        try:
            self.fsm.force_idle()
        except Exception:
            pass
            
        log.info("Shutdown complete.")

    # ── CLI loop ──────────────────────────────────────────────────────────────

    async def run_cli(self) -> None:
        """Main CLI input loop. Runs until 'quit' or KeyboardInterrupt."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                user_input = await loop.run_in_executor(None, self._prompt)
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            await self._handle_command(user_input)

    def _prompt(self) -> str:
        state = self.fsm.state.name
        from colorama import Fore, Style
        return input(f"\n{Fore.BLUE}[{state}]{Style.RESET_ALL} You: ")

    async def _handle_command(self, text: str) -> None:
        lower = text.lower().strip()

        if lower in ("quit", "exit", "q"):
            self._running = False
            return

        if lower == "help":
            self._print_help()
            return

        if lower == "status":
            await self._cmd_status()
            return

        if lower == "memory":
            self._cmd_memory()
            return

        if lower == "history":
            self._cmd_history()
            return

        if lower == "reset":
            self._cmd_reset()
            return

        if lower.startswith("vision "):
            image_path = text[7:].strip()
            await self._cmd_vision(image_path)
            return

        # Default: treat as intent → plan
        await self._cmd_plan(text)

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _cmd_plan(self, intent: str) -> None:
        # Check if FSM allows transitioning to PLANNING
        try:
            self.fsm.transition(State.PLANNING)
        except IllegalTransitionError as exc:
            print(f"❌ State error: Cannot plan right now ({exc}). Type 'reset' if stuck.")
            return

        print("🤔 Planning...")
        loop = asyncio.get_event_loop()

        try:
            plan, risk = await loop.run_in_executor(None, self.plan_for_intent, intent)
        except Exception as exc:
            log.error(f"Planning failed: {exc}", exc_info=True)
            try:
                self.fsm.transition(State.ERROR)
            except IllegalTransitionError:
                self.fsm.force_idle()
            print(f"❌ Planning error: {exc}")
            return

        self._print_plan(plan, risk)

        # Handle risk outcomes
        if risk.is_blocked:
            print(f"\n🚫 Plan BLOCKED — forbidden action detected.")
            self.fsm.force_idle()
            return

        # In V1/V2 execution is blocked regardless, but we simulate confirmation for high-risk
        if risk.level.name in ["HIGH", "MEDIUM"]:
            confirmed = await self._confirm_plan(plan, risk)
            if not confirmed:
                print("❌ Plan aborted by user.")
                self.fsm.force_idle() # Safely return to IDLE, not ABORTED
                return

        print("\n✅ Plan reviewed. (Execution is strictly blocked until V3.)")
        self.fsm.force_idle()

    async def _confirm_plan(self, plan: dict, risk) -> bool:
        """Ask user to confirm a medium/high risk plan."""
        from colorama import Fore, Style
        print(f"\n{Fore.YELLOW}⚠  Risk Level: {risk.level.name}{Style.RESET_ALL}")
        
        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None, lambda: input("\nProceed? [y/N]: ").strip().lower()
            )
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    async def _cmd_status(self) -> None:
        from colorama import Fore, Style
        print(f"\n{'='*50}")
        print(f"  Jarvis v{self._config.get('general', 'version', fallback='2.0.0')}")
        print(f"{'='*50}")
        print(f"  FSM State:    {Fore.GREEN}{self.fsm.state.name}{Style.RESET_ALL}")
        
        try:
            print(f"  Ollama:       {'✅ online' if self.planner.ping() else '❌ offline'}")
        except AttributeError:
            pass
            
        print(f"  Voice:        {'✅ active' if self._voice_on else '❌ disabled'}")

        ok, entries, err = logger_mod.verify_audit()
        audit_status = f"✅ OK ({entries} entries)" if ok else f"❌ TAMPERED — {err}"
        print(f"  Audit Log:    {audit_status}")
        print(f"{'='*50}\n")

    def _cmd_memory(self) -> None:
        # Fallback for displaying memory in CLI
        try:
            # Assumes a sqlite pass-through method exists for debugging
            print("To view semantic memory, check ChromaDB. (CLI Memory view limited in V2).")
        except Exception as e:
            print(f"Memory print error: {e}")

    def _cmd_history(self) -> None:
        """Show recent audit log entries."""
        audit_path = self._config.get("logging", "audit_file", fallback="logs/audit.jsonl")
        path = Path(audit_path)
        if not path.exists():
            print("No audit log yet.")
            return
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-20:]
        print(f"\n── Last {len(recent)} audit entries ──")
        for line in recent:
            try:
                entry = json.loads(line)
                print(f"  [{entry['ts'][11:19]}] {entry['event']}  {json.dumps(entry.get('payload', {}))[:80]}")
            except Exception:
                print(f"  {line[:100]}")
        print()

    def _cmd_reset(self) -> None:
        try:
            self.fsm.force_idle()
            print("✅ Reset to IDLE.")
            self.audit("MANUAL_RESET", {})
        except Exception as exc:
            print(f"⚠  {exc}")

    async def _cmd_vision(self, image_path: str) -> None:
        if not image_path:
            print("Usage: vision <path>")
            return
        print(f"🔍 Analyzing image: {image_path}")
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self.vision.analyze, image_path)
            print(f"\n📷 Vision result:\n{result}\n")
            self.audit("VISION_ANALYZE", {"path": image_path, "result_len": len(result)})
        except (FileNotFoundError, ValueError) as exc:
            print(f"❌ {exc}")
        except Exception as exc:
            print(f"❌ Vision error: {exc}")

    # ── Print helpers ─────────────────────────────────────────────────────────

    def _print_plan(self, plan: dict, risk) -> None:
        from colorama import Fore, Style
        print(f"\n{'─'*50}")
        print(f"📋 Plan: {plan.get('summary', '(no summary)')}")
        print(f"   Confidence: {plan.get('confidence', 0.0):.0%}  |  Risk: {risk.level.name}")

        steps = plan.get("steps", [])
        if steps:
            print("   Steps:")
            for s in steps:
                print(f"     {s.get('id', '?')}. [{s.get('action', '?')}] {s.get('description', '')}")

        if plan.get("clarification_needed"):
            print(f"\n   ❓ {plan.get('clarification_prompt', '')}")
        print(f"{'─'*50}")

    def _print_banner(self) -> None:
        from colorama import Fore, Style
        v = self._config.get("general", "version", fallback="2.0.0")
        mode = "VOICE + CLI" if self._voice_on else "CLI only"
        print(f"""
{Fore.BLUE}╔══════════════════════════════════════╗
║          JARVIS  v{v}            ║
║  Offline · Local · Human-in-the-loop ║
╚══════════════════════════════════════╝{Style.RESET_ALL}
Mode: {mode}
Type 'help' for commands, 'quit' to exit.
""")

    def _print_help(self) -> None:
        print("""
Commands:
  <any text>        Generate a plan for your intent
  vision <path>     Analyze an image with LLaVA
  status            Show system status
  memory            Show stored facts
  history           Show recent audit log
  reset             Reset from ERROR/ABORTED
  help              Show this help
  quit              Shutdown
""")