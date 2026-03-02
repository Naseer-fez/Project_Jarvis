"""Wake-word detection with continuous-listen fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pvporcupine
except ImportError:  # optional dependency
    pvporcupine = None  # type: ignore[assignment]

try:
    from pvrecorder import PvRecorder
except ImportError:  # optional dependency
    PvRecorder = None  # type: ignore[assignment]


class WakeWordDetector:
    """Waits for wake word if available, else behaves as continuous mode."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.wake_word = self._get("wake_word", "jarvis").lower().strip() or "jarvis"
        self.access_key = os.environ.get("PORCUPINE_ACCESS_KEY", self._get("porcupine_access_key", "")).strip()
        self._stop_event = threading.Event()

        self._continuous_mode = pvporcupine is None or PvRecorder is None
        if self._continuous_mode:
            logger.warning("pvporcupine/pvrecorder unavailable - using continuous listening fallback")

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    async def wait_for_wake(self) -> bool:
        """Return when wake word is detected, or immediately in fallback mode."""
        if self._continuous_mode:
            await asyncio.sleep(0.1)
            return not self._stop_event.is_set()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._wait_blocking)

    def _wait_blocking(self) -> bool:
        porcupine = None
        recorder = None

        try:
            kwargs = {"keywords": [self.wake_word]}
            if self.access_key:
                kwargs["access_key"] = self.access_key

            try:
                porcupine = pvporcupine.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Wake-word init failed (%s). Falling back to continuous mode.", exc)
                self._continuous_mode = True
                return not self._stop_event.is_set()

            recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
            recorder.start()

            while not self._stop_event.is_set():
                pcm = recorder.read()
                if porcupine.process(pcm) >= 0:
                    return True

            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wake-word detection failed (%s). Falling back to continuous mode.", exc)
            self._continuous_mode = True
            return not self._stop_event.is_set()
        finally:
            if recorder is not None:
                try:
                    recorder.stop()
                    recorder.delete()
                except Exception:
                    pass
            if porcupine is not None:
                try:
                    porcupine.delete()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop_event.set()


__all__ = ["WakeWordDetector"]
