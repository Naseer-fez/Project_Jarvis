"""
core/voice/tts.py - Text-to-speech output layer.
"""

from __future__ import annotations

import logging
import re
import threading

log = logging.getLogger("jarvis.voice.tts")

try:
    from colorama import Fore, Style
except ImportError:
    class _DummyColor:
        CYAN = ""
        RESET_ALL = ""

    Fore = _DummyColor()
    Style = _DummyColor()

_SENT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


class TTS:
    def __init__(self, config) -> None:
        self._voice = config.get("voice", "tts_voice", fallback="en_US-lessac-medium")
        self._streaming = config.getboolean("voice", "tts_streaming", fallback=True)
        self._cli_fallback = config.getboolean("voice", "tts_fallback_cli", fallback=True)

        self._stop_event = threading.Event()
        self._speaking = False
        self._lock = threading.Lock()
        self._backend = self._init_backend()

    def _init_backend(self) -> str:
        try:
            from piper import PiperVoice

            self._piper_voice = PiperVoice.load(self._voice)
            log.info(f"TTS backend: Piper ({self._voice})")
            return "piper"
        except ImportError:
            log.warning("piper-tts not installed - trying pyttsx3 fallback")
        except Exception as exc:
            log.warning(f"Piper TTS init failed ({exc}) - trying pyttsx3 fallback")

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            self._pyttsx3_engine = engine
            log.info("TTS backend: pyttsx3")
            return "pyttsx3"
        except Exception:
            pass

        log.warning("No TTS audio backend - using CLI fallback only")
        return "cli"

    def speak(self, text: str) -> None:
        if not text.strip():
            return

        with self._lock:
            self._stop_event.clear()
            self._speaking = True

        try:
            if self._streaming and self._backend != "cli":
                for sentence in _split_sentences(text):
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
        self._stop_event.set()
        log.debug("TTS stop requested")

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def _speak_chunk(self, text: str) -> None:
        if self._backend == "piper":
            self._speak_piper(text)
        elif self._backend == "pyttsx3":
            self._speak_pyttsx3(text)
        else:
            self._speak_cli(text)

    def _speak_piper(self, text: str) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            audio_stream = self._piper_voice.synthesize_stream_raw(text)
            for audio_bytes in audio_stream:
                if self._stop_event.is_set():
                    break
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype("float32") / 32768.0
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
        print(f"\n{Fore.CYAN}Jarvis:{Style.RESET_ALL} {text}")
