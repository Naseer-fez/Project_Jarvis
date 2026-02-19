"""
core/voice/voice_loop.py — Async voice loop orchestrator.

Wires wake word → STT → planner → TTS into the V1 state machine.

Invariants:
  - V1 FSM is the single source of truth for state
  - Voice loop can only SUBMIT tasks to the controller — never execute them
  - All voice events are written to the audit log
  - Voice loop crash → CLI fallback, no hang
  - Cancel word → immediate IDLE from any voice state
  - Listen timeout → return to IDLE if no speech within listen_timeout_s

State sequence:
  IDLE
    └── [wake word] → LISTENING
          └── [speech captured] → TRANSCRIBING
                └── [transcript ready] → PLANNING
                      └── [plan ready] → SPEAKING
                            └── [done] → IDLE

  From any voice state:
    cancel word   → IDLE
    timeout       → IDLE
    error         → ERROR (auto-recover to IDLE)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional, TYPE_CHECKING

from core.state_machine import State, IllegalTransitionError

if TYPE_CHECKING:
    from core.controller import Controller

log = logging.getLogger("jarvis.voice.loop")


class VoiceLoop:
    """
    Async voice loop that orchestrates the full wake→listen→transcribe→plan→speak cycle.

    Must be created AFTER the controller is initialised.
    Call start() to launch, stop() for graceful shutdown.
    """

    def __init__(self, controller: "Controller", config) -> None:
        self._ctrl   = controller
        self._config = config
        self._loop   = asyncio.get_event_loop()

        # Timeouts
        self._listen_timeout = float(config.get("voice", "listen_timeout_s", fallback="8"))

        # Events
        self._wake_event   = asyncio.Event()
        self._cancel_event = asyncio.Event()
        self._stop_flag    = False

        # Components (initialised in start())
        self._wake_detector = None
        self._stt           = None
        self._tts           = None

        self._task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise components and launch the async loop task."""
        from core.voice.wake_word import WakeWordDetector
        from core.voice.stt import STT
        from core.voice.tts import TTS

        self._stt = STT(self._config)
        self._tts = TTS(self._config)

        self._wake_detector = WakeWordDetector(
            config=self._config,
            loop=self._loop,
            on_wake=self._on_wake_word,
            on_cancel=self._on_cancel_word,
        )
        self._wake_detector.start()

        self._task = asyncio.ensure_future(self._run())
        log.info("Voice loop started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._stop_flag = True
        self._wake_event.set()
        if self._wake_detector:
            self._wake_detector.stop()
        if self._tts:
            self._tts.stop()
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        log.info("Voice loop stopped")

    # ── Callbacks (called from wake word thread via call_soon_threadsafe) ─────

    def _on_wake_word(self) -> None:
        """Called from wake word detector thread."""
        self._wake_event.set()

    def _on_cancel_word(self) -> None:
        """Called from wake word detector thread."""
        self._cancel_event.set()
        if self._tts:
            self._tts.stop()  # interrupt speech immediately

    # ── Main coroutine ────────────────────────────────────────────────────────

    async def _run(self) -> None:
        """Main voice loop. Runs until stop_flag is set."""
        log.info("Voice loop: waiting for wake word...")
        while not self._stop_flag:
            try:
                await self._wait_for_wake()
                if self._stop_flag:
                    break
                await self._voice_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error(f"Voice loop error: {exc}", exc_info=True)
                self._ctrl.audit("VOICE_LOOP_ERROR", {"error": str(exc)})
                await self._recover_to_idle()
                await asyncio.sleep(1.0)

    async def _wait_for_wake(self) -> None:
        """Block until wake word is detected."""
        self._wake_event.clear()
        self._cancel_event.clear()
        await self._wake_event.wait()

    async def _voice_cycle(self) -> None:
        """One full wake → listen → transcribe → plan → speak cycle."""
        fsm = self._ctrl.fsm

        # ── 1. Transition to LISTENING ────────────────────────────────────────
        try:
            fsm.transition(State.LISTENING)
        except IllegalTransitionError as exc:
            log.warning(f"Cannot enter LISTENING: {exc}")
            return

        self._ctrl.audit("VOICE_WAKE", {"state": "LISTENING"})
        self._tts_print("🎤 Listening...")

        # ── 2. Capture audio with timeout and cancel check ────────────────────
        self._cancel_event.clear()
        transcript_result = await asyncio.wait_for(
            self._capture_with_cancel(),
            timeout=self._listen_timeout + 5.0,  # outer hard timeout
        ) if not self._cancel_event.is_set() else None

        if self._cancel_event.is_set() or transcript_result is None:
            log.info("Voice cycle cancelled or timed out")
            await self._recover_to_idle()
            return

        # ── 3. Transition to TRANSCRIBING ─────────────────────────────────────
        try:
            fsm.transition(State.TRANSCRIBING)
        except IllegalTransitionError:
            await self._recover_to_idle()
            return

        self._ctrl.audit("VOICE_TRANSCRIPT", {
            "text":       transcript_result.text,
            "audio_hash": transcript_result.audio_hash,
            "duration_s": transcript_result.duration_s,
            "confidence": transcript_result.confidence,
        })

        transcript_text = transcript_result.text
        log.info(f"Transcript: \"{transcript_text}\"")

        # Check for cancel word in transcript
        cancel_words = [
            w.strip().lower()
            for w in self._config.get("voice", "cancel_words", fallback="cancel,stop").split(",")
        ]
        if any(cw in transcript_text.lower() for cw in cancel_words):
            log.info("Cancel word in transcript — aborting cycle")
            await self._recover_to_idle()
            return

        # ── 4. Plan ───────────────────────────────────────────────────────────
        try:
            fsm.transition(State.PLANNING)
        except IllegalTransitionError:
            await self._recover_to_idle()
            return

        # Delegate to controller's plan handler (reuses V1 logic)
        plan, risk = await asyncio.get_event_loop().run_in_executor(
            None, self._ctrl.plan_for_intent, transcript_text
        )

        self._ctrl.audit("VOICE_PLAN", {
            "intent": transcript_text,
            "plan_summary": plan.get("summary", ""),
            "risk_level": risk.level.name,
            "blocked": risk.is_blocked,
        })

        # ── 5. Risk check ─────────────────────────────────────────────────────
        if risk.is_blocked:
            response = f"I can't do that. {risk.summary()}"
            log.warning(f"Plan blocked: {risk.summary()}")
        elif plan.get("clarification_needed"):
            response = plan.get("clarification_prompt", "I'm not sure how to help with that.")
        else:
            response = plan.get("summary", "Plan generated — check the screen for details.")

        # ── 6. Speak ──────────────────────────────────────────────────────────
        try:
            fsm.transition(State.SPEAKING)
        except IllegalTransitionError:
            await self._recover_to_idle()
            return

        self._ctrl.audit("VOICE_SPEAK", {"text": response})

        # Run TTS in executor so it doesn't block the event loop
        await asyncio.get_event_loop().run_in_executor(
            None, self._tts.speak, response
        )

        # ── 7. Return to IDLE ─────────────────────────────────────────────────
        try:
            fsm.transition(State.IDLE)
        except IllegalTransitionError:
            await self._recover_to_idle()
            return

        log.info("Voice cycle complete — back to IDLE")

    async def _capture_with_cancel(self):
        """Run STT capture in executor, cancellable via cancel_event."""
        loop = asyncio.get_event_loop()

        # Run blocking STT capture in thread pool
        stt_task = loop.run_in_executor(None, self._stt.capture_and_transcribe)

        # Also watch for cancel
        cancel_watch = asyncio.ensure_future(self._watch_cancel())

        done, pending = await asyncio.wait(
            [stt_task, cancel_watch],
            timeout=self._listen_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

        if self._cancel_event.is_set():
            return None

        if stt_task in done:
            return stt_task.result()

        # Timeout
        log.info("Listen timeout — no speech detected")
        return None

    async def _watch_cancel(self) -> None:
        """Coroutine that completes when cancel_event is set."""
        while not self._cancel_event.is_set():
            await asyncio.sleep(0.05)

    async def _recover_to_idle(self) -> None:
        """Force FSM back to IDLE from any voice state."""
        fsm = self._ctrl.fsm
        voice_states = {State.LISTENING, State.TRANSCRIBING, State.SPEAKING, State.PLANNING}
        try:
            if fsm.state in voice_states:
                fsm.force_idle()
            elif fsm.state == State.ERROR:
                fsm.reset()
        except IllegalTransitionError as exc:
            log.error(f"Recovery failed: {exc}")

        self._wake_event.clear()
        self._cancel_event.clear()
        log.debug("Recovered to IDLE")

    def _tts_print(self, text: str) -> None:
        """Print to console (not spoken — status messages only)."""
        from colorama import Fore, Style
        print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")
