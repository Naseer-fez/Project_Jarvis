"""
core/voice/tts.py
-----------------
Local Text-to-Speech using Piper TTS.
Generates speech offline; plays via sounddevice.

Piper model files (.onnx + .onnx.json) must be in:
    D:\\AI\\Jarvis\\data\\piper\\<voice_model>\\

Download voices from:
    https://huggingface.co/rhasspy/piper-voices
"""

import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Wraps Piper TTS for fully local voice synthesis.

    Piper runs as a subprocess: text -> WAV bytes -> numpy array -> sounddevice playback.
    This keeps the main thread unblocked during generation (async_speak).
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                piper_model_dir      (str) Directory containing .onnx voice models
                voice_model          (str) e.g. 'en_US-lessac-medium'
                speaker_id           (int) 0 for single-speaker models
                output_device_index  (int) -1 for default playback device
        """
        self.model_dir = Path(config.get("piper_model_dir", r"D:\AI\Jarvis\data\piper"))
        self.voice_model = config.get("voice_model", "en_US-lessac-medium")
        self.speaker_id = int(config.get("speaker_id", 0))
        self.output_device = int(config.get("output_device_index", -1))

        self._model_path: Optional[Path] = None
        self._playback_lock = threading.Lock()
        self._current_playback: Optional[threading.Thread] = None

        self._resolve_model()

    def _resolve_model(self):
        """Locate the .onnx model file in the model directory."""
        voice_dir = self.model_dir / self.voice_model
        candidates = list(voice_dir.glob("*.onnx")) if voice_dir.exists() else []

        if not candidates:
            flat = self.model_dir / f"{self.voice_model}.onnx"
            if flat.exists():
                self._model_path = flat
                logger.info("Piper model: %s", flat)
                return

            logger.warning(
                "Piper model '%s' not found in %s.\n"
                "Download from https://huggingface.co/rhasspy/piper-voices\n"
                "and place the .onnx + .onnx.json files in:\n  %s",
                self.voice_model, self.model_dir, voice_dir,
            )
            self._model_path = None
            return

        self._model_path = candidates[0]
        logger.info("Piper model resolved: %s", self._model_path)

    def _synthesize_wav(self, text: str) -> Optional[Path]:
        """
        Call Piper subprocess to synthesize text -> WAV file.

        Returns:
            Path to temporary WAV file, or None on failure.
        """
        if self._model_path is None:
            logger.error("No Piper model loaded. Cannot synthesize speech.")
            return None

        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            tmp_path = Path(tmp.name)

            cmd = [
                "piper",
                "--model", str(self._model_path),
                "--speaker", str(self.speaker_id),
                "--output_file", str(tmp_path),
            ]

            process = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )

            if process.returncode != 0:
                logger.error("Piper error: %s", process.stderr.decode())
                tmp_path.unlink(missing_ok=True)
                return None

            return tmp_path

        except subprocess.TimeoutExpired:
            logger.error("Piper TTS timed out for text: '%s'", text[:50])
            return None
        except FileNotFoundError:
            logger.error(
                "Piper executable not found. Install it and ensure it's on PATH.\n"
                "See: https://github.com/rhasspy/piper/releases"
            )
            return None
        except Exception as e:
            logger.error("Piper synthesis error: %s", e, exc_info=True)
            return None

    def _play_wav(self, wav_path: Path):
        """Load WAV and play via sounddevice (blocking within its own thread)."""
        try:
            with self._playback_lock:
                data, samplerate = sf.read(str(wav_path), dtype="float32")
                device = self.output_device if self.output_device >= 0 else None
                sd.play(data, samplerate=samplerate, device=device)
                sd.wait()
        except Exception as e:
            logger.error("Audio playback error: %s", e, exc_info=True)
        finally:
            wav_path.unlink(missing_ok=True)

    def speak(self, text: str):
        """
        Synthesize and play text synchronously (blocks until done).
        Use async_speak for non-blocking behavior.
        """
        logger.info("[TTS] Speaking: '%s'", text[:80])
        wav_path = self._synthesize_wav(text)
        if wav_path:
            self._play_wav(wav_path)

    def async_speak(self, text: str):
        """
        Synthesize and play text in a background thread.
        Returns immediately; use is_speaking() to check state.
        """
        logger.info("[TTS] Async speaking: '%s'", text[:80])

        def _run():
            wav_path = self._synthesize_wav(text)
            if wav_path:
                self._play_wav(wav_path)

        self._current_playback = threading.Thread(target=_run, name="TTSThread", daemon=True)
        self._current_playback.start()

    def is_speaking(self) -> bool:
        """Returns True if async TTS playback is in progress."""
        return self._current_playback is not None and self._current_playback.is_alive()

    def wait_until_done(self):
        """Block until current async_speak finishes."""
        if self._current_playback:
            self._current_playback.join()

    def stop(self):
        """Interrupt current playback immediately."""
        try:
            sd.stop()
        except Exception:
            pass
