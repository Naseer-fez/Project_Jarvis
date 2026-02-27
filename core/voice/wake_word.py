"""
core/voice/wake_word.py
-----------------------
Local wake word detection using Picovoice Porcupine.
Runs in its own thread; signals main loop via threading.Event.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Wraps pvporcupine + pvrecorder for fully local wake word detection.

    Usage:
        detector = WakeWordDetector(config)
        detector.start(callback=on_wake)   # non-blocking
        ...
        detector.stop()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                access_key       (str)   Porcupine AccessKey
                keyword_path     (str)   Path to .ppn keyword file
                sensitivity      (float) 0.0–1.0
                microphone_index (int)   -1 for default
        """
        self.access_key = config["access_key"]
        self.keyword_path = Path(config["keyword_path"])
        self.sensitivity = float(config.get("sensitivity", 0.5))
        self.mic_index = int(config.get("microphone_index", -1))

        self._porcupine = None
        self._recorder = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callback = None

        self._validate_keyword_path()

    def _validate_keyword_path(self):
        if not self.keyword_path.exists():
            raise FileNotFoundError(
                f"Porcupine keyword file not found: {self.keyword_path}\n"
                "Download it from https://console.picovoice.ai/ and place it in "
                "D:\\AI\\Jarvis\\data\\porcupine\\"
            )

    def _init_porcupine(self):
        try:
            import pvporcupine
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[str(self.keyword_path)],
                sensitivities=[self.sensitivity],
            )
            logger.info(
                "Porcupine initialized | frame_length=%d sample_rate=%d",
                self._porcupine.frame_length,
                self._porcupine.sample_rate,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Porcupine: {e}") from e

    def _init_recorder(self):
        try:
            from pvrecorder import PvRecorder
            device_index = self.mic_index if self.mic_index >= 0 else -1
            self._recorder = PvRecorder(
                frame_length=self._porcupine.frame_length,
                device_index=device_index,
            )
            logger.info("PvRecorder initialized | device_index=%d", device_index)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PvRecorder: {e}") from e

    def _listen_loop(self):
        """Background thread: continuously read audio frames and check for wake word."""
        try:
            self._recorder.start()
            logger.info("Wake word detection active — say 'Jarvis' to activate.")

            while not self._stop_event.is_set():
                pcm_frame = self._recorder.read()
                result = self._porcupine.process(pcm_frame)

                if result >= 0:
                    logger.info("Wake word detected! (keyword index=%d)", result)
                    if self._callback:
                        self._callback()

        except Exception as e:
            logger.error("Wake word listener error: %s", e, exc_info=True)
        finally:
            if self._recorder:
                self._recorder.stop()

    def start(self, callback=None):
        """
        Start wake word detection in a background daemon thread.

        Args:
            callback: Callable invoked (from background thread) when wake word fires.
                      Should be thread-safe (e.g., set a threading.Event).
        """
        if self._thread and self._thread.is_alive():
            logger.warning("Wake word detector already running.")
            return

        self._callback = callback
        self._stop_event.clear()

        self._init_porcupine()
        self._init_recorder()

        self._thread = threading.Thread(
            target=self._listen_loop,
            name="WakeWordThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and clean up resources."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

        if self._recorder:
            try:
                self._recorder.delete()
            except Exception:
                pass
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass

        logger.info("Wake word detector stopped.")

    @staticmethod
    def list_audio_devices():
        """Utility: print all available audio input devices with their indices."""
        try:
            from pvrecorder import PvRecorder
            devices = PvRecorder.get_available_devices()
            print("Available audio input devices:")
            for i, device in enumerate(devices):
                print(f"  [{i}] {device}")
        except ImportError:
            print("pvrecorder not installed. Run: pip install pvrecorder")
