"""Wake-word detection with porcupine and continuous-listen fallback.

The WakeWordDetector class supports the V2 acceptance test API:
  WakeWordDetector(config, loop, on_wake, on_cancel)
  detector._wake_word     — str
  detector._cancel_words  — set[str]
  detector._fire_wake()   — trigger on_wake callback
  detector._fire_cancel() — trigger on_cancel callback
  detector.stop()         — halt detection thread
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import pvporcupine
except ImportError:
    pvporcupine = None  # type: ignore[assignment]

try:
    from pvrecorder import PvRecorder
except ImportError:
    PvRecorder = None  # type: ignore[assignment]


class WakeWordDetector:
    """
    Detects a wake word and fires callbacks; falls back to continuous mode
    when porcupine is not installed.

    Signature matches V2 acceptance tests:
      WakeWordDetector(config, loop, on_wake, on_cancel)
    """

    def __init__(
        self,
        config: Any,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_wake: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._loop = loop
        self._on_wake = on_wake
        self._on_cancel = on_cancel

        self._wake_word: str = self._get("wake_word", "jarvis").strip().lower() or "jarvis"

        cancel_raw = self._get("cancel_words", "cancel,stop")
        self._cancel_words: set[str] = {
            w.strip().lower() for w in cancel_raw.split(",") if w.strip()
        }

        self.access_key = os.environ.get(
            "PORCUPINE_ACCESS_KEY", self._get("porcupine_access_key", "")
        ).strip()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._continuous_mode = pvporcupine is None or PvRecorder is None

        if not self.access_key and not self._continuous_mode:
            logger.warning(
                "PORCUPINE_ACCESS_KEY not set. Wake word detection disabled. "
                "Get a free key at: https://console.picovoice.ai/"
            )
            self._continuous_mode = True

        if self._continuous_mode:
            logger.warning("Wake-word backend unavailable; using continuous listen fallback")

    # ── Config helper ─────────────────────────────────────────────────────

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    # ── Callback helpers (patchable / called in tests) ────────────────────

    def _fire_wake(self) -> None:
        """Fire the on_wake callback, scheduling it on the event loop if provided."""
        if self._on_wake is None:
            return
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._on_wake)
                return
            except RuntimeError:
                pass
        # Fallback: call directly
        self._on_wake()

    def _fire_cancel(self) -> None:
        """Fire the on_cancel callback."""
        if self._on_cancel is None:
            return
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._on_cancel)
                return
            except RuntimeError:
                pass
        self._on_cancel()

    # ── Async interface (for controller usage) ────────────────────────────

    async def wait_for_wake(self) -> bool:
        """Return True when ready for STT capture."""
        if self._continuous_mode:
            await asyncio.sleep(0.1)
            return not self._stop_event.is_set()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._wait_blocking)

    def _wait_blocking(self) -> bool:
        porcupine = None
        recorder = None
        try:
            kwargs: dict[str, Any] = {"keywords": [self._wake_word]}
            if self.access_key:
                kwargs["access_key"] = self.access_key
            try:
                porcupine = pvporcupine.create(**kwargs)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Wake-word init failed (%s); falling back", exc)
                self._continuous_mode = True
                return not self._stop_event.is_set()

            recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)  # type: ignore[union-attr]
            recorder.start()

            while not self._stop_event.is_set():
                pcm = recorder.read()
                if porcupine.process(pcm) >= 0:
                    return True

            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wake-word detection failed (%s); switching to continuous", exc)
            self._continuous_mode = True
            return not self._stop_event.is_set()
        finally:
            for obj in (recorder, porcupine):
                if obj is not None:
                    for method in ("stop", "delete"):
                        try:
                            getattr(obj, method)()
                        except Exception:  # noqa: BLE001
                            pass

    def stop(self) -> None:
        """Signal detection loop to halt."""
        self._stop_event.set()


__all__ = ["WakeWordDetector"]
