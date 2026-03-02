"""Text-to-speech with fallback chain: edge-tts -> pyttsx3 -> print-to-stdout.

The TTS class provides the synchronous API expected by the V2 acceptance tests:
  - TTS(config)
  - tts.speak(text)          — synchronous
  - tts.stop()               — interrupt ongoing speech
  - tts.is_speaking          — bool property
  - tts._backend             — "edge", "pyttsx3", or "cli"
  - tts._stop_event          — threading.Event
  - tts._init_backend(...)   — callable, returns backend name (patchable in tests)

The TextToSpeech class is the async variant kept for backward compat.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pyttsx3  # type: ignore[import]
except ImportError:
    pyttsx3 = None  # type: ignore[assignment]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS output."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class TTS:
    """
    Synchronous text-to-speech with a three-tier fallback chain.

    Backend priority: edge-tts  →  pyttsx3  →  CLI (print to stdout)
    All backend errors are caught silently; the next fallback is tried.

    Design decisions:
      * _init_backend() is a separate method so tests can monkeypatch it.
      * _stop_event is a threading.Event that halts sentence-by-sentence output.
      * speak() is blocking (runs in the calling thread) so tests can assert on
        stdout without races.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._stop_event = threading.Event()
        self._speaking_lock = threading.Lock()
        self._is_speaking = False

        self._backend: str = self._init_backend(config)

    # ── Backend selection (patchable) ─────────────────────────────────────

    def _init_backend(self, config: Any) -> str:
        """Detect the best available backend. Returns 'pyttsx3', 'edge', or 'cli'."""
        # Try pyttsx3 first (works offline)
        if pyttsx3 is not None:
            try:
                eng = pyttsx3.init()
                eng.stop()
                return "pyttsx3"
            except Exception:  # noqa: BLE001
                pass
        # CLI / print fallback — always available
        return "cli"

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    # ── Public API ────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak *text* synchronously, respecting stop_event between sentences."""
        text = (text or "").strip()
        if not text:
            return

        self._stop_event.clear()
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]

        with self._speaking_lock:
            self._is_speaking = True
            try:
                for sentence in sentences:
                    if self._stop_event.is_set():
                        break
                    self._speak_sentence(sentence)
            finally:
                self._is_speaking = False

    def stop(self) -> None:
        """Request interruption of ongoing speech."""
        self._stop_event.set()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _speak_sentence(self, sentence: str) -> None:
        """Speak a single sentence using the configured backend."""
        if self._backend == "pyttsx3" and pyttsx3 is not None:
            try:
                eng = pyttsx3.init()
                eng.say(sentence)
                eng.runAndWait()
                return
            except Exception as exc:  # noqa: BLE001
                logger.debug("pyttsx3 speech failed: %s", exc)

        # CLI fallback — always works
        print(f"Jarvis: {sentence}")


# ── Async variant (kept for backward compat with controller_v2) ───────────────

try:
    import asyncio
    import os
    import tempfile
    from pathlib import Path as _Path

    try:
        import edge_tts as _edge_tts  # type: ignore[import]
    except ImportError:
        _edge_tts = None  # type: ignore[assignment]

    class TextToSpeech:
        """Async TTS wrapper used by the newer async controller path."""

        def __init__(self, config: Any) -> None:
            self._config = config
            self.preferred_engine = self._get("tts_engine", "edge-tts").strip().lower()
            self.voice_name = self._get("tts_voice", "en-US-GuyNeural")
            self._pyttsx3_engine = None

            if pyttsx3 is not None:
                try:
                    self._pyttsx3_engine = pyttsx3.init()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pyttsx3 init failed: %s", exc)

        def _get(self, key: str, default: str) -> str:
            try:
                return str(self._config.get("voice", key, fallback=default))
            except Exception:  # noqa: BLE001
                return default

        async def speak(self, text: str) -> None:  # type: ignore[override]
            text = (text or "").strip()
            if not text:
                return
            for engine in self._engine_chain():
                if engine == "edge" and await self._speak_edge(text):
                    return
                if engine == "pyttsx3" and await self._speak_pyttsx3(text):
                    return
            print(f"Jarvis: {text}")

        def _engine_chain(self) -> list[str]:
            if self.preferred_engine in {"edge", "edge-tts"}:
                return ["edge", "pyttsx3", "print"]
            if self.preferred_engine in {"pyttsx3", "offline"}:
                return ["pyttsx3", "edge", "print"]
            return ["edge", "pyttsx3", "print"]

        async def _speak_edge(self, text: str) -> bool:
            if _edge_tts is None:
                return False
            try:
                tmp = _Path(tempfile.gettempdir()) / f"jarvis_tts_{abs(hash(text))}.mp3"
                com = _edge_tts.Communicate(text=text, voice=self.voice_name)
                await com.save(str(tmp))
                if os.name == "nt":
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, os.startfile, str(tmp))  # type: ignore[attr-defined]
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("edge-tts failed: %s", exc)
            return False

        async def _speak_pyttsx3(self, text: str) -> bool:
            if self._pyttsx3_engine is None:
                return False
            def _run() -> bool:
                try:
                    self._pyttsx3_engine.say(text)
                    self._pyttsx3_engine.runAndWait()
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pyttsx3 speak failed: %s", exc)
                    return False
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _run)

except Exception:  # noqa: BLE001
    # Minimal stub if imports fail
    class TextToSpeech:  # type: ignore[no-redef]
        def __init__(self, config: Any) -> None:
            pass
        async def speak(self, text: str) -> None:
            print(f"Jarvis: {text}")


__all__ = ["TTS", "TextToSpeech", "_split_sentences"]
