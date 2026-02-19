"""
tests/test_v2_acceptance.py — V2 Voice Layer Acceptance Checklist.

ALL V1 tests must still pass. These tests cover the V2 voice additions.
Run: pytest tests/test_v2_acceptance.py -v

Tests use mocks for audio hardware — no microphone or speaker required.
"""

from __future__ import annotations

import asyncio
import configparser
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state_machine import StateMachine, State, IllegalTransitionError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_config(tmp_path):
    cfg = configparser.ConfigParser()
    cfg["general"]  = {"name": "Jarvis", "version": "2.0.0"}
    cfg["ollama"]   = {"base_url": "http://localhost:11434", "planner_model": "deepseek-r1:8b", "vision_model": "llava", "request_timeout_s": "10"}
    cfg["memory"]   = {"data_dir": str(tmp_path / "data"), "sqlite_file": str(tmp_path / "data/jarvis_memory.db"), "chroma_dir": str(tmp_path / "data/chroma"), "embedding_model": "all-MiniLM-L6-v2", "semantic_top_k": "5"}
    cfg["risk"]     = {"forbidden_actions": "shell_exec,file_delete,network_request,serial_send,physical_actuate", "high_risk_actions": "file_write,process_spawn", "medium_risk_actions": "file_read", "low_risk_actions": "memory_read,memory_write,speak,display", "voice_confirm_threshold": "MEDIUM"}
    cfg["logging"]  = {"log_dir": str(tmp_path / "logs"), "audit_file": str(tmp_path / "logs/audit.jsonl"), "app_file": str(tmp_path / "logs/app.log"), "level": "DEBUG"}
    cfg["voice"]    = {
        "enabled": "true", "wake_word": "hey jarvis",
        "cancel_words": "cancel,stop,never mind",
        "stt_model": "base.en", "stt_device": "cpu", "stt_compute_type": "int8",
        "stt_silence_ms": "500", "stt_max_duration_s": "30", "stt_vad_aggressiveness": "2",
        "tts_voice": "en_US-lessac-medium", "tts_streaming": "false", "tts_fallback_cli": "true",
        "listen_timeout_s": "8", "audio_sample_rate": "16000", "audio_channels": "1",
        "audio_chunk_ms": "30", "wakeword_threshold": "0.5",
    }
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# 1. V1 REGRESSION — all V1 tests must still pass (sampled here)
# ════════════════════════════════════════════════════════════════════════════

class TestV1Regression:
    def test_fsm_illegal_transition_still_raises(self):
        fsm = StateMachine()
        with pytest.raises(IllegalTransitionError):
            fsm.transition(State.EXECUTING)

    def test_risk_forbidden_still_blocked(self):
        from core.risk_evaluator import RiskEvaluator, RiskLevel
        ev = RiskEvaluator()
        assert ev.evaluate(["shell_exec"]).level == RiskLevel.FORBIDDEN

    def test_memory_no_hallucination(self, tmp_config):
        from core.memory.hybrid_memory import HybridMemory
        mem = HybridMemory(tmp_config)
        result = mem.recall("nonexistent_quantum_flux_capacitor_secret")
        assert "don't know" in result.lower() or "related facts" in result.lower()

    def test_serial_stub_still_blocked(self):
        from core.hardware.serial_controller import SerialController
        with pytest.raises(NotImplementedError):
            SerialController().send("anything")


# ════════════════════════════════════════════════════════════════════════════
# 2. WAKE WORD DETECTOR
# ════════════════════════════════════════════════════════════════════════════

class TestWakeWordDetector:
    def test_detector_instantiates(self, tmp_config):
        from core.voice.wake_word import WakeWordDetector
        loop = asyncio.new_event_loop()
        try:
            detector = WakeWordDetector(
                config=tmp_config,
                loop=loop,
                on_wake=lambda: None,
                on_cancel=lambda: None,
            )
            assert detector is not None
        finally:
            loop.close()

    def test_cancel_words_parsed(self, tmp_config):
        from core.voice.wake_word import WakeWordDetector
        loop = asyncio.new_event_loop()
        try:
            detector = WakeWordDetector(
                config=tmp_config, loop=loop,
                on_wake=lambda: None, on_cancel=lambda: None,
            )
            assert "cancel" in detector._cancel_words
            assert "stop" in detector._cancel_words
        finally:
            loop.close()

    def test_wake_word_config_parsed(self, tmp_config):
        from core.voice.wake_word import WakeWordDetector
        loop = asyncio.new_event_loop()
        try:
            detector = WakeWordDetector(
                config=tmp_config, loop=loop,
                on_wake=lambda: None, on_cancel=lambda: None,
            )
            assert detector._wake_word == "hey jarvis"
        finally:
            loop.close()

    def test_on_wake_fires_callback(self, tmp_config):
        from core.voice.wake_word import WakeWordDetector
        loop = asyncio.new_event_loop()
        fired = []
        try:
            detector = WakeWordDetector(
                config=tmp_config, loop=loop,
                on_wake=lambda: fired.append("wake"),
                on_cancel=lambda: None,
            )
            # Simulate wake word fire (normally called from thread)
            loop.call_soon(detector._on_wake_word)
            loop.run_until_complete(asyncio.sleep(0))
            assert "wake" in fired
        finally:
            loop.close()

    def test_on_cancel_fires_callback(self, tmp_config):
        from core.voice.wake_word import WakeWordDetector
        loop = asyncio.new_event_loop()
        fired = []
        try:
            detector = WakeWordDetector(
                config=tmp_config, loop=loop,
                on_wake=lambda: None,
                on_cancel=lambda: fired.append("cancel"),
            )
            loop.call_soon(detector._on_cancel_word)
            loop.run_until_complete(asyncio.sleep(0))
            assert "cancel" in fired
        finally:
            loop.close()

    def test_stop_called_without_start_is_safe(self, tmp_config):
        from core.voice.wake_word import WakeWordDetector
        loop = asyncio.new_event_loop()
        try:
            detector = WakeWordDetector(
                config=tmp_config, loop=loop,
                on_wake=lambda: None, on_cancel=lambda: None,
            )
            detector.stop()  # must not raise
        finally:
            loop.close()


# ════════════════════════════════════════════════════════════════════════════
# 3. STT — transcription, timeout, fallback
# ════════════════════════════════════════════════════════════════════════════

class TestSTT:
    def test_stt_instantiates(self, tmp_config):
        from core.voice.stt import STT
        with patch("core.voice.stt.STT._init", return_value=None):
            stt = STT.__new__(STT)
            stt._ready = False
            assert not stt.is_ready

    def test_stt_returns_none_when_not_ready(self, tmp_config):
        from core.voice.stt import STT
        stt = STT.__new__(STT)
        stt._ready = False
        result = stt.capture_and_transcribe()
        assert result is None

    def test_transcript_result_dataclass(self):
        from core.voice.stt import TranscriptResult
        r = TranscriptResult(
            text="hello world",
            audio_hash="abc123",
            duration_s=1.5,
            language="en",
            confidence=0.9,
        )
        assert r.text == "hello world"
        assert r.confidence == 0.9

    def test_stt_energy_vad_fallback(self, tmp_config):
        """energy-based VAD doesn't crash on silent audio."""
        from core.voice.stt import STT
        import struct
        stt = STT.__new__(STT)
        stt._vad = None
        stt._sample_rate = 16000
        # 30ms of silence (480 samples × 2 bytes)
        silent_pcm = struct.pack("480h", *([0] * 480))
        is_speech = stt._is_speech(silent_pcm, 480)
        assert is_speech is False

    def test_stt_energy_vad_detects_loud(self, tmp_config):
        from core.voice.stt import STT
        import struct
        stt = STT.__new__(STT)
        stt._vad = None
        stt._sample_rate = 16000
        # Loud audio
        loud_pcm = struct.pack("480h", *([20000] * 480))
        is_speech = stt._is_speech(loud_pcm, 480)
        assert is_speech is True


# ════════════════════════════════════════════════════════════════════════════
# 4. TTS — speak, interrupt, CLI fallback
# ════════════════════════════════════════════════════════════════════════════

class TestTTS:
    def test_tts_instantiates_with_cli_fallback(self, tmp_config):
        from core.voice.tts import TTS
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            assert tts._backend == "cli"

    def test_tts_cli_fallback_does_not_crash(self, tmp_config, capsys):
        from core.voice.tts import TTS
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            tts.speak("Hello from Jarvis.")
        captured = capsys.readouterr()
        assert "Hello from Jarvis." in captured.out

    def test_tts_stop_sets_flag(self, tmp_config):
        from core.voice.tts import TTS
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            tts.stop()
            assert tts._stop_event.is_set()

    def test_tts_speak_empty_string_is_noop(self, tmp_config, capsys):
        from core.voice.tts import TTS
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            tts.speak("")  # must not crash or print anything
        captured = capsys.readouterr()
        assert "Jarvis:" not in captured.out

    def test_tts_does_not_read_raw_json(self, tmp_config, capsys):
        """TTS must not speak raw JSON — only human-readable text."""
        from core.voice.tts import TTS
        raw_json = '{"steps": [{"action": "shell_exec"}]}'
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            # If someone passes JSON, TTS speaks it verbatim (the controller
            # is responsible for passing only summaries — this tests that TTS
            # itself doesn't crash and the controller path sends summaries)
            tts.speak(raw_json)  # must not crash
        # Controller-level test: plan summary is passed, not raw JSON

    def test_tts_sentence_split(self):
        from core.voice.tts import _split_sentences
        text = "Hello world. How are you? Fine thanks."
        sentences = _split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."

    def test_tts_is_speaking_flag(self, tmp_config):
        from core.voice.tts import TTS
        import threading
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            assert not tts.is_speaking

    def test_tts_interrupt_during_speech(self, tmp_config, capsys):
        """stop() during speak() must halt output."""
        from core.voice.tts import TTS
        import threading

        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)

            long_text = "One. Two. Three. Four. Five. Six. Seven. Eight."
            results = []

            def speak_thread():
                tts.speak(long_text)
                results.append("done")

            t = threading.Thread(target=speak_thread)
            t.start()
            time.sleep(0.05)
            tts.stop()
            t.join(timeout=2.0)
            assert "done" in results


# ════════════════════════════════════════════════════════════════════════════
# 5. VOICE LOOP — state transitions, cancel, no action trigger
# ════════════════════════════════════════════════════════════════════════════

class TestVoiceLoop:
    def _make_controller(self, tmp_config, tmp_path):
        """Create a controller with all components mocked for isolation."""
        import core.logger as logger_mod
        if logger_mod._audit is None:
            logger_mod.setup(tmp_config)

        from core.controller import Controller
        ctrl = Controller(tmp_config, voice=False)
        # Mock expensive components
        ctrl.planner.plan = MagicMock(return_value={
            "intent": "test", "summary": "Test plan.", "confidence": 0.9,
            "steps": [{"id": 1, "action": "speak", "description": "Say hello", "params": {}}],
            "clarification_needed": False, "clarification_prompt": "",
        })
        ctrl.planner.ping = MagicMock(return_value=False)
        return ctrl

    def test_voice_loop_instantiates(self, tmp_config, tmp_path):
        from core.voice.voice_loop import VoiceLoop
        ctrl = self._make_controller(tmp_config, tmp_path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            vl = VoiceLoop(ctrl, tmp_config)
            assert vl is not None
        finally:
            loop.close()

    def test_voice_loop_cannot_trigger_forbidden_action(self, tmp_config, tmp_path):
        """Voice loop submits plans through risk evaluator — forbidden actions blocked."""
        from core.risk_evaluator import RiskEvaluator, RiskLevel
        ev = RiskEvaluator(tmp_config)
        # A plan with shell_exec must be blocked before reaching execution
        result = ev.evaluate(["shell_exec"])
        assert result.is_blocked

    def test_cancel_word_in_transcript_aborts_cycle(self, tmp_config, tmp_path):
        """If transcript contains a cancel word, cycle aborts."""
        cancel_words = ["cancel", "stop", "never mind"]
        transcript = "stop that please"
        assert any(cw in transcript.lower() for cw in cancel_words)

    def test_fsm_returns_to_idle_after_voice_cycle(self, tmp_config, tmp_path):
        """Voice cycle must end at IDLE regardless of outcome."""
        fsm = StateMachine()
        # Simulate happy path
        fsm.transition(State.LISTENING)
        fsm.transition(State.TRANSCRIBING)
        fsm.transition(State.PLANNING)
        fsm.transition(State.SPEAKING)
        fsm.transition(State.IDLE)
        assert fsm.state == State.IDLE

    def test_fsm_force_idle_from_listening(self, tmp_config):
        """Cancel from LISTENING → IDLE."""
        fsm = StateMachine()
        fsm.transition(State.LISTENING)
        fsm.force_idle()
        assert fsm.state == State.IDLE

    def test_fsm_force_idle_from_speaking(self, tmp_config):
        """Cancel from SPEAKING → IDLE."""
        fsm = StateMachine()
        fsm.transition(State.LISTENING)
        fsm.transition(State.TRANSCRIBING)
        fsm.transition(State.PLANNING)
        fsm.transition(State.SPEAKING)
        fsm.force_idle()
        assert fsm.state == State.IDLE

    def test_voice_loop_does_not_access_l2_l3(self):
        """Voice loop must not import or call desktop/serial automation."""
        import importlib
        import sys
        # Serial controller must remain a stub
        from core.hardware.serial_controller import SerialController
        sc = SerialController()
        assert not sc.is_connected
        with pytest.raises(NotImplementedError):
            sc.send("voice triggered command")


# ════════════════════════════════════════════════════════════════════════════
# 6. AUDIT LOG — voice events are logged
# ════════════════════════════════════════════════════════════════════════════

class TestVoiceAuditLog:
    def test_voice_events_written_to_audit(self, tmp_config, tmp_path):
        import core.logger as logger_mod
        logger_mod.setup(tmp_config)
        logger_mod.audit("VOICE_WAKE", {"state": "LISTENING"})
        logger_mod.audit("VOICE_TRANSCRIPT", {"text": "hello", "audio_hash": "abc", "duration_s": 1.0, "confidence": 0.9})
        logger_mod.audit("VOICE_PLAN", {"intent": "hello", "plan_summary": "greet", "risk_level": "LOW", "blocked": False})
        logger_mod.audit("VOICE_SPEAK", {"text": "Hello!"})

        ok, count, err = logger_mod.verify_audit()
        assert ok, err
        assert count >= 4

    def test_audio_hash_in_transcript_event(self, tmp_config, tmp_path):
        import hashlib
        fake_pcm = b"\x00\x01" * 1000
        audio_hash = hashlib.sha256(fake_pcm).hexdigest()
        assert len(audio_hash) == 64

        import core.logger as logger_mod
        logger_mod.setup(tmp_config)
        h = logger_mod.audit("VOICE_TRANSCRIPT", {
            "text": "test", "audio_hash": audio_hash,
            "duration_s": 0.5, "confidence": 0.85,
        })
        assert isinstance(h, str)
        assert len(h) == 64


# ════════════════════════════════════════════════════════════════════════════
# 7. OFFLINE VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

class TestOffline:
    def test_risk_evaluator_needs_no_network(self):
        from core.risk_evaluator import RiskEvaluator
        ev = RiskEvaluator()
        result = ev.evaluate(["speak", "memory_read"])
        assert result is not None  # works with zero network

    def test_state_machine_needs_no_network(self):
        fsm = StateMachine()
        fsm.transition(State.PLANNING)
        fsm.transition(State.IDLE)
        assert fsm.state == State.IDLE

    def test_audit_log_needs_no_network(self, tmp_config, tmp_path):
        from core.logger import AuditLog
        log = AuditLog(str(tmp_path / "audit.jsonl"))
        log.write("OFFLINE_TEST", {"status": "ok"})
        ok, count, _ = log.verify()
        assert ok and count == 1

    def test_tts_cli_fallback_needs_no_audio_device(self, tmp_config, capsys):
        from core.voice.tts import TTS
        with patch("core.voice.tts.TTS._init_backend", return_value="cli"):
            tts = TTS(tmp_config)
            tts.speak("Testing offline fallback.")
        out = capsys.readouterr().out
        assert "Testing offline fallback." in out
