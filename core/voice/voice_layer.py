"""
core/voice/voice_layer.py  –  Jarvis Voice Layer V2
====================================================
Responsibilities
----------------
1. Wake-word detection via pvporcupine (runs in a daemon thread).
2. Speech-to-text via openai-whisper (local model, runs in executor).
3. Text-to-speech via piper-tts + sounddevice playback (async, interruptible).
4. Barge-in: if the wake word fires while TTS is playing, playback stops
   immediately and Jarvis re-enters listening mode.
5. AutonomyPolicy hook: if the controller signals REQUIRE_CONFIRM the voice
   layer reads the question aloud and listens for yes / no / abort.

Integration contract
--------------------
- Call ``VoiceLayer(config, text_handler)`` where ``text_handler`` is an
  async coroutine  ``async def handler(text: str) -> str``.
- Call ``await voice_layer.start()`` once from your async entry-point.
- Call ``await voice_layer.stop()`` on shutdown.
- To ask a confirmation question from the agentic layer call
  ``await voice_layer.ask_confirm(question: str) -> bool | None``
  which returns True (yes), False (no), or None (abort / timeout).

All audio device indexes are auto-detected (system default) unless explicitly
set in jarvis.ini.  No cloud APIs are used.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import struct
import threading
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

logger = logging.getLogger("jarvis.voice")

# ---------------------------------------------------------------------------
# Optional-import guards – give friendly errors if libraries are missing
# ---------------------------------------------------------------------------

def _require(pkg: str, install: str):
    """Raise ImportError with an actionable message."""
    raise ImportError(
        f"'{pkg}' is required for the voice layer.\n"
        f"Install it with:  {install}"
    )


try:
    import pvporcupine                        # wake-word
except ImportError:
    pvporcupine = None  # type: ignore

try:
    import pyaudio                            # mic capture
except ImportError:
    pyaudio = None  # type: ignore

try:
    import whisper as openai_whisper          # STT
except ImportError:
    openai_whisper = None  # type: ignore

try:
    from piper import PiperVoice              # TTS
except ImportError:
    PiperVoice = None  # type: ignore

try:
    import sounddevice as sd                  # TTS playback
except ImportError:
    sd = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIRM_KEYWORDS_YES   = {"yes", "yeah", "yep", "confirm", "affirmative", "do it", "proceed"}
CONFIRM_KEYWORDS_NO    = {"no", "nope", "negative", "deny", "reject", "don't", "stop"}
CONFIRM_KEYWORDS_ABORT = {"abort", "cancel", "quit", "nevermind", "never mind"}

SILENCE_THRESHOLD      = 300    # RMS below this = silence
SILENCE_DURATION_SEC   = 1.5    # stop recording after this many seconds of silence
MAX_RECORD_SEC         = 30     # hard cap on utterance length
CONFIRM_TIMEOUT_SEC    = 15     # how long to wait for a yes/no before returning None


# ---------------------------------------------------------------------------
# VoiceConfig  (thin wrapper around jarvis.ini [voice] section)
# ---------------------------------------------------------------------------

class VoiceConfig:
    def __init__(self, cfg: ConfigParser):
        v = cfg["voice"] if cfg.has_section("voice") else {}

        self.enabled: bool          = cfg.getboolean("voice", "enabled", fallback=False)
        self.porcupine_key: str     = v.get("porcupine_access_key", "")
        self.porcupine_keyword: str = v.get("porcupine_keyword", "jarvis")
        self.porcupine_model: str   = v.get("porcupine_model_path", "")   # "" → built-in

        self.whisper_model: str     = v.get("whisper_model", "base.en")
        self.whisper_model_dir: str = v.get("whisper_model_dir", "data/whisper")

        self.piper_model: str       = v.get(
            "piper_model_path", "data/voices/en_US-lessac-medium.onnx"
        )
        self.piper_config: str      = v.get(
            "piper_config_path", "data/voices/en_US-lessac-medium.onnx.json"
        )

        # -1 means "use system default"
        self.input_device: int      = int(v.get("input_device_index",  "-1"))
        self.output_device: int     = int(v.get("output_device_index", "-1"))

        self.stt_language: str      = v.get("stt_language", "en")
        self.sample_rate: int       = int(v.get("sample_rate", "16000"))


# ---------------------------------------------------------------------------
# TTS engine (Piper, synchronous, called from executor)
# ---------------------------------------------------------------------------

class _TTSEngine:
    """Wraps PiperVoice; generates raw PCM bytes."""

    def __init__(self, cfg: VoiceConfig):
        if PiperVoice is None:
            _require("piper-tts", "pip install piper-tts")

        model_path = Path(cfg.piper_model)
        config_path = Path(cfg.piper_config)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {model_path}\n"
                "Download from https://huggingface.co/rhasspy/piper-voices"
            )

        self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._sample_rate = cfg.sample_rate
        logger.info("Piper TTS loaded: %s", model_path.name)

    def synthesize(self, text: str) -> bytes:
        """Return raw 16-bit mono PCM bytes at self._sample_rate."""
        buf = io.BytesIO()
        with io.BytesIO() as wav_buf:
            self._voice.synthesize(text, wav_buf)
            wav_buf.seek(0)
            # Piper writes a wav; strip the 44-byte header to get raw PCM
            wav_buf.read(44)
            return wav_buf.read()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


# ---------------------------------------------------------------------------
# STT engine (Whisper, synchronous, called from executor)
# ---------------------------------------------------------------------------

class _STTEngine:
    def __init__(self, cfg: VoiceConfig):
        if openai_whisper is None:
            _require("openai-whisper", "pip install openai-whisper")

        model_name = cfg.whisper_model
        model_dir  = cfg.whisper_model_dir or None
        logger.info("Loading Whisper model '%s' …", model_name)
        self._model = openai_whisper.load_model(model_name, download_root=model_dir)
        self._language = cfg.stt_language
        logger.info("Whisper ready.")

    def transcribe(self, audio_np: np.ndarray, sample_rate: int) -> str:
        """Transcribe float32 mono numpy array; return lower-cased text."""
        # Whisper expects float32 at 16 kHz
        if sample_rate != 16000:
            import librosa  # optional up-/down-sample
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)

        result = self._model.transcribe(
            audio_np,
            language=self._language,
            fp16=False,
        )
        text = result.get("text", "").strip()
        logger.debug("STT result: %r", text)
        return text


# ---------------------------------------------------------------------------
# Audio capture helpers (synchronous, run in threads)
# ---------------------------------------------------------------------------

def _rms(chunk: bytes) -> float:
    """Root-mean-square of a 16-bit PCM chunk."""
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0


class _MicRecorder:
    """Records a single utterance (blocking).  Returns numpy float32 array."""

    def __init__(self, cfg: VoiceConfig, pa: "pyaudio.PyAudio"):
        self._pa        = pa
        self._sr        = cfg.sample_rate
        self._device    = cfg.input_device if cfg.input_device >= 0 else None
        self._frames_per_buf = 1024

    def record_utterance(self) -> np.ndarray:
        """Block until the user stops speaking; return float32 mono array."""
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sr,
            input=True,
            input_device_index=self._device,
            frames_per_buffer=self._frames_per_buf,
        )
        frames: list[bytes] = []
        silent_chunks = 0
        silence_limit  = int(SILENCE_DURATION_SEC * self._sr / self._frames_per_buf)
        max_chunks     = int(MAX_RECORD_SEC       * self._sr / self._frames_per_buf)

        logger.debug("Recording utterance …")
        try:
            for _ in range(max_chunks):
                chunk = stream.read(self._frames_per_buf, exception_on_overflow=False)
                frames.append(chunk)
                if _rms(chunk) < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= silence_limit:
                        break
                else:
                    silent_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()

        raw = b"".join(frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio


# ---------------------------------------------------------------------------
# VoiceLayer  – the public API
# ---------------------------------------------------------------------------

class VoiceLayer:
    """
    Orchestrates wake-word → STT → controller → TTS in a non-blocking way.

    Parameters
    ----------
    config : ConfigParser
        Loaded jarvis.ini.
    text_handler : async callable (str) -> str
        Async function that receives transcribed text and returns Jarvis's
        text response.  Typically wraps ``controller_v2.process_input``.
    """

    def __init__(
        self,
        config: ConfigParser,
        text_handler: Callable[[str], Awaitable[str]],
    ):
        self._cfg     = VoiceConfig(config)
        self._handler = text_handler
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Shared state
        self._tts_playing   = threading.Event()   # set while TTS audio plays
        self._barge_in      = threading.Event()   # set when wake-word fires mid-TTS
        self._shutdown      = threading.Event()
        self._confirm_queue: "queue.Queue[Optional[str]]" = queue.Queue()

        # Lazy-loaded engines (heavy; loaded once on start())
        self._tts: Optional[_TTSEngine]  = None
        self._stt: Optional[_STTEngine]  = None
        self._pa:  Optional["pyaudio.PyAudio"] = None
        self._recorder: Optional[_MicRecorder] = None

        # Wake-word thread
        self._wake_thread: Optional[threading.Thread] = None

        # Playback stream reference for barge-in interrupt
        self._current_playback: Optional["sd.OutputStream"] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load models and start the wake-word listener thread."""
        if not self._cfg.enabled:
            logger.info("Voice layer disabled in config; skipping start.")
            return

        self._loop = asyncio.get_running_loop()

        if pyaudio is None:
            _require("pyaudio", "pip install pyaudio")
        if sd is None:
            _require("sounddevice", "pip install sounddevice")

        logger.info("Initialising voice layer …")
        self._pa       = pyaudio.PyAudio()
        self._recorder = _MicRecorder(self._cfg, self._pa)

        # Load STT and TTS in thread-pool to avoid blocking the event loop
        self._stt = await self._loop.run_in_executor(None, lambda: _STTEngine(self._cfg))
        self._tts = await self._loop.run_in_executor(None, lambda: _TTSEngine(self._cfg))

        # Start wake-word thread
        self._wake_thread = threading.Thread(
            target=self._wake_word_loop, daemon=True, name="jarvis-wake"
        )
        self._wake_thread.start()
        logger.info("Voice layer started.  Listening for wake word '%s' …",
                    self._cfg.porcupine_keyword)

    async def stop(self) -> None:
        """Gracefully stop the voice layer."""
        self._shutdown.set()
        if self._pa:
            self._pa.terminate()
        logger.info("Voice layer stopped.")

    # ------------------------------------------------------------------
    # Public: confirmation hook (called by AutonomyPolicy integration)
    # ------------------------------------------------------------------

    async def ask_confirm(self, question: str) -> Optional[bool]:
        """
        Speak ``question``, listen for yes / no / abort.

        Returns
        -------
        True   – user confirmed
        False  – user denied
        None   – abort or timeout
        """
        if not self._cfg.enabled or self._stt is None or self._tts is None:
            # Fall back to console input
            ans = input(f"[CONFIRM] {question} (yes/no/abort): ").strip().lower()
            return _parse_confirm(ans)

        await self.speak(question)

        # Drain stale answers
        while not self._confirm_queue.empty():
            self._confirm_queue.get_nowait()

        # Record and transcribe with a timeout
        try:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, self._recorder.record_utterance)
            text  = await loop.run_in_executor(
                None, lambda: self._stt.transcribe(audio, self._cfg.sample_rate)
            )
            return _parse_confirm(text.lower())
        except Exception as exc:
            logger.error("Confirmation STT failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public: speak a text response (async, interruptible)
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Synthesise ``text`` and play it; can be interrupted by barge-in."""
        if not self._cfg.enabled or self._tts is None:
            print(f"[Jarvis]: {text}")
            return

        loop = asyncio.get_running_loop()
        pcm  = await loop.run_in_executor(None, lambda: self._tts.synthesize(text))

        self._barge_in.clear()
        self._tts_playing.set()
        try:
            await loop.run_in_executor(None, self._play_pcm, pcm)
        finally:
            self._tts_playing.clear()

    # ------------------------------------------------------------------
    # Internal: playback (blocking, runs in executor thread)
    # ------------------------------------------------------------------

    def _play_pcm(self, pcm_bytes: bytes) -> None:
        """Play raw 16-bit PCM.  Stops immediately if barge-in fires."""
        if sd is None:
            return

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sr    = self._tts.sample_rate  # type: ignore[union-attr]

        chunk_size  = sr // 20          # 50 ms chunks
        device_idx  = self._cfg.output_device if self._cfg.output_device >= 0 else None

        try:
            with sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                device=device_idx,
            ) as stream:
                self._current_playback = stream
                offset = 0
                while offset < len(audio):
                    if self._barge_in.is_set() or self._shutdown.is_set():
                        logger.debug("TTS interrupted by barge-in or shutdown.")
                        break
                    chunk = audio[offset : offset + chunk_size]
                    stream.write(chunk)
                    offset += chunk_size
        except Exception as exc:
            logger.error("TTS playback error: %s", exc)
        finally:
            self._current_playback = None

    # ------------------------------------------------------------------
    # Internal: wake-word + main listen loop (daemon thread)
    # ------------------------------------------------------------------

    def _wake_word_loop(self) -> None:
        """Blocking loop; detects wake word then fires the async pipeline."""
        if pvporcupine is None:
            _require("pvporcupine", "pip install pvporcupine")

        try:
            porcupine = self._build_porcupine()
        except Exception as exc:
            logger.error("Failed to initialise Porcupine: %s", exc)
            return

        pa      = pyaudio.PyAudio()
        stream  = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=(
                self._cfg.input_device if self._cfg.input_device >= 0 else None
            ),
            frames_per_buffer=porcupine.frame_length,
        )

        logger.debug("Wake-word listener running (frame_length=%d, sr=%d).",
                     porcupine.frame_length, porcupine.sample_rate)

        try:
            while not self._shutdown.is_set():
                raw   = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm   = struct.unpack_from("h" * porcupine.frame_length, raw)
                index = porcupine.process(pcm)

                if index >= 0:                   # wake word detected
                    logger.info("Wake word detected!")
                    if self._tts_playing.is_set():
                        # Barge-in: interrupt current TTS
                        self._barge_in.set()
                        self._tts_playing.wait(timeout=1.0)

                    # Hand off to async loop
                    if self._loop and not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._handle_utterance(), self._loop
                        )
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()
            logger.debug("Wake-word thread exited.")

    def _build_porcupine(self) -> "pvporcupine.Porcupine":
        """Create the Porcupine instance from config."""
        kwargs: dict = {}

        if self._cfg.porcupine_key:
            kwargs["access_key"] = self._cfg.porcupine_key
        else:
            raise ValueError(
                "porcupine_access_key is required in [voice] config.\n"
                "Get a free key at https://console.picovoice.ai/"
            )

        keyword = self._cfg.porcupine_keyword.lower()
        # Map well-known built-in keywords
        builtin_map = {
            "jarvis":     pvporcupine.KEYWORD_PATHS.get("jarvis"),
            "hey google": pvporcupine.KEYWORD_PATHS.get("hey google"),
            "ok google":  pvporcupine.KEYWORD_PATHS.get("ok google"),
            "hey siri":   pvporcupine.KEYWORD_PATHS.get("hey siri"),
            "alexa":      pvporcupine.KEYWORD_PATHS.get("alexa"),
        }

        if self._cfg.porcupine_model:
            # Custom .ppn model file
            kwargs["keyword_paths"] = [self._cfg.porcupine_model]
        elif keyword in builtin_map and builtin_map[keyword]:
            kwargs["keywords"] = [keyword]
        else:
            logger.warning(
                "No built-in keyword for '%s'. "
                "Falling back to 'jarvis' built-in.", keyword
            )
            kwargs["keywords"] = ["jarvis"]

        return pvporcupine.create(**kwargs)

    # ------------------------------------------------------------------
    # Internal: utterance pipeline (async)
    # ------------------------------------------------------------------

    async def _handle_utterance(self) -> None:
        """Record → STT → handler → TTS."""
        loop = asyncio.get_running_loop()

        # 1. Record
        try:
            audio = await loop.run_in_executor(None, self._recorder.record_utterance)
        except Exception as exc:
            logger.error("Recording failed: %s", exc)
            return

        if audio is None or len(audio) == 0:
            logger.warning("Empty audio captured; skipping.")
            return

        # 2. STT
        try:
            text = await loop.run_in_executor(
                None, lambda: self._stt.transcribe(audio, self._cfg.sample_rate)  # type: ignore
            )
        except Exception as exc:
            logger.error("STT failed: %s", exc)
            return

        if not text:
            logger.debug("Empty transcription; ignoring.")
            return

        logger.info("User said: %r", text)

        # 3. Handler (controller)
        try:
            response = await self._handler(text)
        except Exception as exc:
            logger.error("Controller raised: %s", exc)
            response = "I encountered an error processing that request."

        # 4. TTS
        await self.speak(response or "I'm not sure how to respond to that.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_confirm(text: str) -> Optional[bool]:
    """Map a transcribed yes/no/abort string to True/False/None."""
    words = set(text.lower().split())
    if words & CONFIRM_KEYWORDS_ABORT:
        return None
    if words & CONFIRM_KEYWORDS_YES:
        return True
    if words & CONFIRM_KEYWORDS_NO:
        return False
    # Single-word catch
    if "yes" in text:
        return True
    if "no" in text:
        return False
    return None   # ambiguous → treat as abort
