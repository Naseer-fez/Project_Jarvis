"""
core/voice/stt.py
-----------------
Local Speech-to-Text using OpenAI Whisper (runs fully offline).
Handles audio capture via PyAudio with dynamic energy thresholding
and silence detection — no recording forever.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # PyAudio frames per buffer


class SpeechToText:
    """
    Records audio after wake word trigger, transcribes with Whisper.

    The recorder uses a two-phase approach:
      1. Wait for audio energy above threshold (speech onset).
      2. Stop after `silence_timeout` seconds of low energy (speech end).
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                model_size              (str)   e.g. 'base.en'
                whisper_cache_dir       (str)   D:\\AI\\Jarvis\\data\\whisper
                language                (str)   'en'
                record_timeout_seconds  (float) hard cap on recording duration
                silence_threshold_secs  (float) silence duration to stop recording
                energy_multiplier       (float) RMS multiplier to detect speech
                sample_rate             (int)   16000
                channels                (int)   1
                microphone_index        (int)   -1 for default
        """
        self.model_size = config.get("model_size", "base.en")
        self.cache_dir = Path(config.get("whisper_cache_dir", r"D:\AI\Jarvis\data\whisper"))
        self.language = config.get("language", "en")
        self.record_timeout = float(config.get("record_timeout_seconds", 8))
        self.silence_timeout = float(config.get("silence_threshold_seconds", 1.5))
        self.energy_multiplier = float(config.get("energy_multiplier", 2.5))
        self.sample_rate = int(config.get("sample_rate", WHISPER_SAMPLE_RATE))
        self.channels = int(config.get("channels", 1))
        self.mic_index = int(config.get("microphone_index", -1))

        self._model = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tell Whisper where to cache model weights (keep off C:)
        os.environ["XDG_CACHE_HOME"] = str(self.cache_dir.parent)
        os.environ["WHISPER_CACHE_DIR"] = str(self.cache_dir)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper
            logger.info("Loading Whisper model '%s' (may take a moment)...", self.model_size)
            self._model = whisper.load_model(
                self.model_size,
                download_root=str(self.cache_dir),
            )
            logger.info("Whisper model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_size}': {e}") from e

    def _calibrate_ambient_rms(self, stream, num_chunks: int = 20) -> float:
        """Sample ambient noise to set a dynamic energy threshold."""
        frames = []
        for _ in range(num_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            frames.append(audio_np)

        ambient = np.concatenate(frames)
        rms = float(np.sqrt(np.mean(ambient ** 2))) + 1e-6
        threshold = rms * self.energy_multiplier
        logger.debug("Ambient RMS=%.1f | Speech threshold=%.1f", rms, threshold)
        return threshold

    def _record_audio(self) -> Optional[np.ndarray]:
        """
        Open microphone, calibrate energy, record until silence or timeout.

        Returns:
            numpy float32 array at 16kHz, or None on failure.
        """
        try:
            import pyaudio
        except ImportError:
            raise RuntimeError("pyaudio not installed. Run: pip install pyaudio")

        pa = pyaudio.PyAudio()
        stream = None

        try:
            device_index = self.mic_index if self.mic_index >= 0 else None
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
            )

            logger.info("Calibrating ambient noise...")
            threshold = self._calibrate_ambient_rms(stream)
            logger.info("Listening... (threshold=%.0f, timeout=%.1fs)", threshold, self.record_timeout)

            frames = []
            speech_started = False
            silence_start: Optional[float] = None
            record_start = time.monotonic()

            while True:
                elapsed = time.monotonic() - record_start
                if elapsed > self.record_timeout:
                    logger.warning("Recording timeout reached (%.1fs).", self.record_timeout)
                    break

                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

                if rms > threshold:
                    speech_started = True
                    silence_start = None
                    frames.append(data)
                elif speech_started:
                    frames.append(data)
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= self.silence_timeout:
                        logger.debug("Silence detected — stopping recording.")
                        break

            if not frames:
                logger.warning("No speech detected.")
                return None

            raw = b"".join(frames)
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np

        except Exception as e:
            logger.error("Audio recording error: %s", e, exc_info=True)
            return None
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pa.terminate()

    def transcribe(self) -> Optional[str]:
        """
        Record from microphone and return transcribed text.

        Returns:
            Transcribed string, or None if recording/transcription failed.
        """
        self._load_model()

        logger.info("[STT] Recording...")
        audio_np = self._record_audio()

        if audio_np is None or len(audio_np) < self.sample_rate * 0.3:
            return None

        logger.info("[STT] Transcribing %d samples (%.1fs)...",
                    len(audio_np), len(audio_np) / self.sample_rate)

        try:
            import whisper
            result = self._model.transcribe(
                audio_np,
                language=self.language,
                fp16=False,
                verbose=False,
            )
            text = result.get("text", "").strip()
            logger.info("[STT] Transcribed: '%s'", text)
            return text if text else None

        except Exception as e:
            logger.error("Whisper transcription error: %s", e, exc_info=True)
            return None
