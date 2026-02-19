"""
core/voice/wake_word.py — Wake word detection.

Primary:  OpenWakeWord (open-source, local, MIT licensed)
Fallback: Simple energy + keyword heuristic (when OpenWakeWord unavailable)

Design invariants:
  - Runs in a dedicated daemon thread — NEVER in the main async loop
  - Only signals via asyncio.Event — never modifies state directly
  - Fires two event types: WAKE (detected keyword) and CANCEL (cancel/stop word)
  - CPU usage < 2% at idle

Authority: L0_OBSERVE
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("jarvis.voice.wake_word")


class WakeWordDetector:
    """
    Listens on the default microphone for a wake word.

    Usage:
        detector = WakeWordDetector(config, loop, on_wake_cb, on_cancel_cb)
        detector.start()
        ...
        detector.stop()
    """

    def __init__(
        self,
        config,
        loop: asyncio.AbstractEventLoop,
        on_wake: Callable[[], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self._config     = config
        self._loop       = loop
        self._on_wake    = on_wake
        self._on_cancel  = on_cancel
        self._running    = False
        self._thread: Optional[threading.Thread] = None

        # Config
        self._wake_word    = config.get("voice", "wake_word",    fallback="hey jarvis").lower()
        self._cancel_words = [
            w.strip().lower()
            for w in config.get("voice", "cancel_words", fallback="cancel,stop").split(",")
        ]
        self._threshold = float(config.get("voice", "wakeword_threshold", fallback="0.5"))
        self._sample_rate = int(config.get("voice", "audio_sample_rate", fallback="16000"))
        self._chunk_ms    = int(config.get("voice", "audio_chunk_ms",    fallback="30"))

        self._backend = self._init_backend()

    def _init_backend(self) -> str:
        try:
            import openwakeword
            from openwakeword.model import Model as OWWModel
            self._oww_model = OWWModel(inference_framework="onnx")
            log.info("Wake word backend: OpenWakeWord")
            return "openwakeword"
        except ImportError:
            log.warning("OpenWakeWord not installed — using energy+STT fallback backend")
        except Exception as exc:
            log.warning(f"OpenWakeWord init failed ({exc}) — using fallback backend")

        try:
            import vosk  # noqa: F401
            log.info("Wake word backend: Vosk fallback")
            return "vosk"
        except ImportError:
            pass

        log.warning("No wake word backend available — using dummy (always fires on audio)")
        return "dummy"

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="wakeword"
        )
        self._thread.start()
        log.info(f"Wake word detector started (backend={self._backend}, word='{self._wake_word}')")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    # ── Detection loop ────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        if self._backend == "openwakeword":
            self._run_openwakeword()
        elif self._backend == "vosk":
            self._run_vosk()
        else:
            self._run_dummy()

    def _fire_wake(self) -> None:
        log.info(f"Wake word detected: '{self._wake_word}'")
        self._loop.call_soon_threadsafe(self._on_wake)

    def _fire_cancel(self) -> None:
        log.info("Cancel word detected")
        self._loop.call_soon_threadsafe(self._on_cancel)

    def _run_openwakeword(self) -> None:
        try:
            import pyaudio
            chunk_size = int(self._sample_rate * self._chunk_ms / 1000)
            pa = pyaudio.PyAudio()
            stream = pa.open(
                rate=self._sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=chunk_size,
            )
            import numpy as np
            while self._running:
                pcm = stream.read(chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(pcm, dtype=np.int16)
                predictions = self._oww_model.predict(audio)

                for model_name, score in predictions.items():
                    keyword = model_name.lower().replace("_", " ")
                    if any(cw in keyword for cw in self._cancel_words) and score > self._threshold:
                        self._fire_cancel()
                        break
                    elif score > self._threshold:
                        self._fire_wake()
                        time.sleep(1.5)  # debounce
                        break

            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as exc:
            log.error(f"OpenWakeWord loop error: {exc}")

    def _run_vosk(self) -> None:
        """Vosk-based keyword detection fallback."""
        try:
            import vosk
            import pyaudio
            import json as _json

            model = vosk.Model(lang="en-us")
            rec = vosk.KaldiRecognizer(model, self._sample_rate)
            chunk_size = int(self._sample_rate * self._chunk_ms / 1000)
            pa = pyaudio.PyAudio()
            stream = pa.open(
                rate=self._sample_rate, channels=1,
                format=pyaudio.paInt16, input=True,
                frames_per_buffer=chunk_size,
            )
            while self._running:
                pcm = stream.read(chunk_size, exception_on_overflow=False)
                if rec.AcceptWaveform(pcm):
                    result = _json.loads(rec.Result()).get("text", "").lower()
                    if any(cw in result for cw in self._cancel_words):
                        self._fire_cancel()
                    elif self._wake_word in result:
                        self._fire_wake()
                        time.sleep(1.5)

            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as exc:
            log.error(f"Vosk wake word loop error: {exc}")

    def _run_dummy(self) -> None:
        """
        Dummy backend — used when no real wake word lib is available.
        Fires WAKE every 60 seconds for testing purposes.
        """
        while self._running:
            time.sleep(60)
            if self._running:
                log.debug("Dummy wake word backend: firing synthetic wake event")
                self._fire_wake()
