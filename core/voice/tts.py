"""
core/voice/tts.py — Text-to-speech via Piper TTS.

Design:
  - Offline ONNX inference (Piper)
  - Sentence-streaming for low latency (first chunk < 400ms target)
  - Interrupt flag: calling stop() cancels playback mid-sentence
  - CLI fallback: if audio device unavailable, prints to console
  - NEVER reads raw JSON aloud — only summarised human text

Authority: L1_INTERPRET (output only — no action triggered)
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

log = logging.getLogger("jarvis.voice.tts")

# Sentence split pattern
_SENT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming."""
    sentences = _SENT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


class TTS:
    """Text-to-speech with interrupt support."""

    def __init__(self, config) -> None:
        self._voice         = config.get("voice", "tts_voice",       fallback="en_US-lessac-medium")
        self._streaming     = config.getboolean("voice", "tts_streaming",   fallback=True)
        self._cli_fallback  = config.getboolean("voice", "tts_fallback_cli", fallback=True)

        self._stop_event = threading.Event()
        self._speaking   = False
        self._lock       = threading.Lock()

        self._backend = self._init_backend()

    def _init_backend(self) -> str:
        try:
            from piper import PiperVoice  # piper-tts package
            self._piper_voice = PiperVoice.load(self._voice)
            log.info(f"TTS backend: Piper ({self._voice})")
            return "piper"
        except ImportError:
            log.warning("piper-tts not installed — trying pyttsx3 fallback")
        except Exception as exc:
            log.warning(f"Piper TTS init failed ({exc}) — trying pyttsx3 fallback")

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            self._pyttsx3_engine = engine
            log.info("TTS backend: pyttsx3")
            return "pyttsx3"
        except Exception:
            pass

        log.warning("No TTS audio backend — using CLI fallback only")
        return "cli"

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Speak text. Blocks until complete or interrupted.
        Respects stop_event for cancellation.
        """
        if not text.strip():
            return

        with self._lock:
            self._stop_event.clear()
            self._speaking = True

        try:
            if self._streaming and self._backend != "cli":
                sentences = _split_sentences(text)
                for sentence in sentences:
                    if self._stop_event.is_set():
                        log.debug("TTS interrupted")
                        break
                    self._speak_chunk(sentence)
            else:
                self._speak_chunk(text)
        finally:
            with self._lock:
                self._speaking = False

    def stop(self) -> None:
        """Interrupt current speech immediately."""
        self._stop_event.set()
        log.debug("TTS stop requested")

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    # ── Backends ──────────────────────────────────────────────────────────────

    def _speak_chunk(self, text: str) -> None:
        if self._backend == "piper":
            self._speak_piper(text)
        elif self._backend == "pyttsx3":
            self._speak_pyttsx3(text)
        else:
            self._speak_cli(text)

    def _speak_piper(self, text: str) -> None:
        try:
            import sounddevice as sd
            import numpy as np

            audio_stream = self._piper_voice.synthesize_stream_raw(text)
            for audio_bytes in audio_stream:
                if self._stop_event.is_set():
                    break
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(audio, samplerate=22050, blocking=True)
        except Exception as exc:
            log.warning(f"Piper speak error: {exc}")
            if self._cli_fallback:
                self._speak_cli(text)

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            if not self._stop_event.is_set():
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
        except Exception as exc:
            log.warning(f"pyttsx3 speak error: {exc}")
            if self._cli_fallback:
                self._speak_cli(text)

    def _speak_cli(self, text: str) -> None:
        """CLI fallback — always available."""
        from colorama import Fore, Style
        print(f"\n{Fore.CYAN}🔊 Jarvis:{Style.RESET_ALL} {text}")
