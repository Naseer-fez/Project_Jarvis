"""
core/voice/voice_loop.py - Async voice loop orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from core.state_machine import IllegalTransitionError, State

if TYPE_CHECKING:
    from core.controller import Controller

try:
    from colorama import Fore, Style
except ImportError:
    class _DummyColor:
        YELLOW = ""
        RESET_ALL = ""

    Fore = _DummyColor()
    Style = _DummyColor()

log = logging.getLogger("jarvis.voice.loop")


class VoiceLoopState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


VoiceStateListener = Callable[[VoiceLoopState], None]


class VoiceLoop:
    def __init__(self, controller: "Controller", config) -> None:
        self._ctrl = controller
        self._config = config
        self._loop = asyncio.get_event_loop()

        self._listen_timeout = float(config.get("voice", "listen_timeout_s", fallback="8"))

        self._wake_event = asyncio.Event()
        self._cancel_event = asyncio.Event()
        self._stop_flag = False

        self._wake_detector = None
        self._stt = None
        self._tts = None
        self._task: Optional[asyncio.Task] = None

        self._yes_words = {"yes", "yeah", "yep", "proceed", "do it", "allow"}
        self._no_words = {"no", "nope", "cancel", "stop", "deny"}

        self._state = VoiceLoopState.IDLE
        self._state_listeners: list[VoiceStateListener] = []

    @property
    def state(self) -> VoiceLoopState:
        return self._state

    def add_state_listener(self, listener: VoiceStateListener) -> None:
        self._state_listeners.append(listener)

    def _set_state(self, state: VoiceLoopState) -> None:
        if state == self._state:
            return
        self._state = state
        for listener in list(self._state_listeners):
            try:
                listener(state)
            except Exception:
                pass
        if hasattr(self._ctrl, "trace"):
            self._ctrl.trace.state(state=state.value, source="voice_loop")
        try:
            self._ctrl.audit("VOICE_LOOP_STATE", {"state": state.value})
        except Exception:
            pass

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        from core.voice.stt import STT
        from core.voice.tts import TTS
        from core.voice.wake_word import WakeWordDetector

        self._stt = STT(self._config)
        self._tts = TTS(self._config)
        self._wake_detector = WakeWordDetector(
            config=self._config,
            loop=self._loop,
            on_wake=self._on_wake_word,
            on_cancel=self._on_cancel_word,
        )
        self._wake_detector.start()
        self._set_state(VoiceLoopState.IDLE)
        self._task = asyncio.ensure_future(self._run())
        log.info("Voice loop started")

    async def stop(self) -> None:
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
        self._set_state(VoiceLoopState.IDLE)
        log.info("Voice loop stopped")

    # -- Wake/cancel callbacks -------------------------------------------

    def _on_wake_word(self) -> None:
        self._wake_event.set()

    def _on_cancel_word(self) -> None:
        self._cancel_event.set()
        self._set_state(VoiceLoopState.INTERRUPTED)
        if self._tts:
            self._tts.stop()
        if hasattr(self._ctrl, "request_interrupt"):
            try:
                self._ctrl.request_interrupt(reason="voice_cancel")
            except Exception:
                pass

    # -- Main loop --------------------------------------------------------

    async def _run(self) -> None:
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
                if hasattr(self._ctrl, "trace"):
                    self._ctrl.trace.error("voice_loop", str(exc))
                await self._recover_to_idle()
                await asyncio.sleep(1.0)

    async def _wait_for_wake(self) -> None:
        self._wake_event.clear()
        self._cancel_event.clear()
        self._set_state(VoiceLoopState.IDLE)
        await self._wake_event.wait()

    async def _voice_cycle(self) -> None:
        fsm = self._ctrl.fsm

        try:
            fsm.transition(State.LISTENING)
        except IllegalTransitionError as exc:
            log.warning(f"Cannot enter LISTENING: {exc}")
            return

        self._set_state(VoiceLoopState.LISTENING)
        self._ctrl.audit("VOICE_WAKE", {"state": "LISTENING"})
        self._tts_print("Listening...")

        self._cancel_event.clear()
        transcript_result = (
            await asyncio.wait_for(
                self._capture_with_cancel(),
                timeout=self._listen_timeout + 5.0,
            )
            if not self._cancel_event.is_set()
            else None
        )
        if self._cancel_event.is_set() or transcript_result is None:
            log.info("Voice cycle cancelled or timed out")
            await self._recover_to_idle()
            return

        try:
            fsm.transition(State.TRANSCRIBING)
        except IllegalTransitionError:
            await self._recover_to_idle()
            return

        self._ctrl.audit(
            "VOICE_TRANSCRIPT",
            {
                "text": transcript_result.text,
                "audio_hash": transcript_result.audio_hash,
                "duration_s": transcript_result.duration_s,
                "confidence": transcript_result.confidence,
            },
        )
        self._emit_transcription(transcript_result.text, is_final=True, confidence=transcript_result.confidence)

        transcript_text = transcript_result.text
        log.info(f'Transcript: "{transcript_text}"')

        cancel_words = [
            w.strip().lower()
            for w in self._config.get("voice", "cancel_words", fallback="cancel,stop").split(",")
        ]
        if any(cw in transcript_text.lower() for cw in cancel_words):
            log.info("Cancel word in transcript - aborting cycle")
            await self._recover_to_idle()
            return

        try:
            fsm.transition(State.PLANNING)
        except IllegalTransitionError:
            await self._recover_to_idle()
            return
        self._set_state(VoiceLoopState.THINKING)

        plan, risk = await asyncio.get_event_loop().run_in_executor(
            None, self._ctrl.plan_for_intent, transcript_text
        )
        self._ctrl.audit(
            "VOICE_PLAN",
            {
                "intent": transcript_text,
                "plan_summary": plan.get("summary", ""),
                "risk_level": risk.level.name,
                "blocked": risk.is_blocked,
            },
        )

        if plan.get("clarification_needed") and not plan.get("steps"):
            response = plan.get("clarification_prompt", "I need clarification before I can proceed.")
        else:
            response, _ = await self._ctrl.execute_plan_with_feedback(
                intent=transcript_text,
                plan=plan,
                risk=risk,
                confirm_callback=self._confirm_voice_execution,
            )

        self._ctrl.audit("VOICE_SPEAK", {"text": response})
        self._set_state(VoiceLoopState.SPEAKING)
        speech_thread = self._tts.speak_async(response, emotion=str(plan.get("speech_emotion", "neutral")))
        while speech_thread.is_alive():
            if self._cancel_event.is_set():
                self._tts.stop()
                break
            await asyncio.sleep(0.05)

        try:
            if fsm.state == State.SPEAKING:
                fsm.transition(State.IDLE)
            elif fsm.state == State.INTERRUPTED:
                fsm.transition(State.IDLE)
            else:
                fsm.force_idle()
        except IllegalTransitionError:
            await self._recover_to_idle()
            return

        self._set_state(VoiceLoopState.IDLE)
        log.info("Voice cycle complete - back to IDLE")

    async def _confirm_voice_execution(self, plan: dict, risk: Any) -> bool:
        del plan
        self._ctrl.audit("VOICE_CONFIRM_REQUEST", {"risk_level": risk.level.name})
        prompt = "This requires permission. Proceed?"
        prompt_thread = self._tts.speak_async(prompt, emotion="serious")
        while prompt_thread.is_alive():
            if self._cancel_event.is_set():
                self._tts.stop()
                return False
            await asyncio.sleep(0.05)

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._stt.capture_and_transcribe),
                timeout=self._listen_timeout + 2.0,
            )
        except asyncio.TimeoutError:
            return False

        if result is None or not result.text:
            return False
        text = result.text.lower().strip()
        if any(w in text for w in self._yes_words):
            return True
        if any(w in text for w in self._no_words):
            return False
        return False

    async def _capture_with_cancel(self):
        loop = asyncio.get_event_loop()

        def _on_partial(text: str, is_final: bool) -> None:
            loop.call_soon_threadsafe(self._emit_transcription, text, is_final, None)

        stt_task = loop.run_in_executor(None, self._stt.capture_and_transcribe_stream, _on_partial)
        cancel_watch = asyncio.ensure_future(self._watch_cancel())

        done, pending = await asyncio.wait(
            [stt_task, cancel_watch],
            timeout=self._listen_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for pending_task in pending:
            pending_task.cancel()

        if self._cancel_event.is_set():
            return None
        if stt_task in done:
            return stt_task.result()

        log.info("Listen timeout - no speech detected")
        return None

    def _emit_transcription(self, text: str, is_final: bool, confidence: Optional[float]) -> None:
        if hasattr(self._ctrl, "trace"):
            self._ctrl.trace.transcription(text=text, is_final=is_final, confidence=confidence)

    async def _watch_cancel(self) -> None:
        while not self._cancel_event.is_set():
            await asyncio.sleep(0.05)

    async def _recover_to_idle(self) -> None:
        fsm = self._ctrl.fsm
        recoverable_states = {
            State.LISTENING,
            State.TRANSCRIBING,
            State.PLANNING,
            State.EXECUTING,
            State.SPEAKING,
            State.INTERRUPTED,
        }
        try:
            if fsm.state in recoverable_states:
                if fsm.state != State.IDLE:
                    fsm.force_idle()
            elif fsm.state == State.ERROR:
                fsm.reset()
        except IllegalTransitionError as exc:
            log.error(f"Recovery failed: {exc}")

        self._wake_event.clear()
        self._cancel_event.clear()
        self._set_state(VoiceLoopState.IDLE)
        log.debug("Recovered to IDLE")

    def _tts_print(self, text: str) -> None:
        print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")
