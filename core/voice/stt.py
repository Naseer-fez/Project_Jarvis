"""
core/voice/stt.py — Speech-to-text using Faster-Whisper.

Design:
  - Captures audio from microphone until silence detected (webrtcvad)
  - Transcribes with faster-whisper (local, offline)
  - Returns plain text transcript + audio SHA-256 hash for audit trail
  - Hard timeout: stops capture after stt_max_duration_s regardless of silence

Authority: L0_OBSERVE (capture + transcribe only — no action triggered)
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("jarvis.voice.stt")

# Silence detection constants
_MIN_SPEECH_FRAMES = 3   # frames of speech required before we start accumulating
_BYTES_PER_SAMPLE  = 2   # 16-bit PCM


@dataclass
class TranscriptResult:
    text: str
    audio_hash: str          # SHA-256 of raw PCM bytes
    duration_s: float
    language: str
    confidence: float        # average segment confidence [0.0–1.0]


class STT:
    """Captures audio and transcribes it with Faster-Whisper."""

    def __init__(self, config) -> None:
        self._model_name   = config.get("voice", "stt_model",          fallback="base.en")
        self._device       = config.get("voice", "stt_device",         fallback="cpu")
        self._compute_type = config.get("voice", "stt_compute_type",   fallback="int8")
        self._sample_rate  = int(config.get("voice", "audio_sample_rate", fallback="16000"))
        self._channels     = int(config.get("voice", "audio_channels",    fallback="1"))
        self._chunk_ms     = int(config.get("voice", "audio_chunk_ms",    fallback="30"))
        self._silence_ms   = int(config.get("voice", "stt_silence_ms",    fallback="500"))
        self._max_dur_s    = int(config.get("voice", "stt_max_duration_s",fallback="30"))
        self._vad_mode     = int(config.get("voice", "stt_vad_aggressiveness", fallback="2"))

        self._model = None
        self._vad   = None
        self._pa    = None
        self._ready = False

        self._init()

    def _init(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
            log.info(f"Whisper model loaded: {self._model_name}")
        except ImportError:
            log.error("faster-whisper not installed — STT unavailable")
            return
        except Exception as exc:
            log.error(f"Whisper model load failed: {exc}")
            return

        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._vad_mode)
        except ImportError:
            log.warning("webrtcvad not installed — using energy-based VAD fallback")

        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
        except Exception as exc:
            log.error(f"PyAudio init failed: {exc}")
            return

        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def capture_and_transcribe(self) -> Optional[TranscriptResult]:
        """
        Open mic, capture speech, transcribe.
        Returns None if STT unavailable or no speech detected.
        """
        if not self._ready:
            log.warning("STT not ready — cannot capture")
            return None

        try:
            pcm_bytes = self._capture_audio()
        except Exception as exc:
            log.error(f"Audio capture failed: {exc}")
            return None

        if not pcm_bytes:
            log.debug("No audio captured")
            return None

        audio_hash = hashlib.sha256(pcm_bytes).hexdigest()
        return self._transcribe(pcm_bytes, audio_hash)

    def _capture_audio(self) -> bytes:
        """Capture until silence or max duration. Returns raw PCM bytes."""
        import pyaudio

        chunk_size = int(self._sample_rate * self._chunk_ms / 1000)
        frames: list[bytes] = []
        silence_frames_needed = int(self._silence_ms / self._chunk_ms)
        max_frames = int(self._max_dur_s * 1000 / self._chunk_ms)

        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

        consecutive_silence = 0
        speech_started = False
        speech_frame_count = 0
        total_frames = 0

        try:
            while total_frames < max_frames:
                pcm = stream.read(chunk_size, exception_on_overflow=False)
                total_frames += 1

                is_speech = self._is_speech(pcm, chunk_size)

                if is_speech:
                    speech_frame_count += 1
                    consecutive_silence = 0
                    if speech_frame_count >= _MIN_SPEECH_FRAMES:
                        speech_started = True
                    if speech_started:
                        frames.append(pcm)
                else:
                    if speech_started:
                        frames.append(pcm)  # include trailing silence frames
                        consecutive_silence += 1
                        if consecutive_silence >= silence_frames_needed:
                            break  # end of utterance
        finally:
            stream.stop_stream()
            stream.close()

        return b"".join(frames)

    def _is_speech(self, pcm: bytes, chunk_size: int) -> bool:
        """VAD or energy-based speech detection."""
        if self._vad is not None:
            try:
                expected = chunk_size * _BYTES_PER_SAMPLE
                if len(pcm) == expected:
                    return self._vad.is_speech(pcm, self._sample_rate)
            except Exception:
                pass

        # Energy-based fallback
        import struct
        samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > 500  # threshold tuned for typical mic levels

    def _transcribe(self, pcm_bytes: bytes, audio_hash: str) -> Optional[TranscriptResult]:
        """Run Faster-Whisper on raw PCM bytes."""
        try:
            import numpy as np
            import soundfile as sf

            # Convert PCM bytes to float32 numpy array
            audio_np = (
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            t0 = time.time()
            segments, info = self._model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True,
            )

            seg_list = list(segments)
            if not seg_list:
                return None

            full_text = " ".join(s.text.strip() for s in seg_list).strip()
            if not full_text:
                return None

            avg_conf = sum(
                max(s.avg_logprob, -5.0) for s in seg_list
            ) / len(seg_list)
            # Convert log-prob to rough 0–1 confidence
            confidence = min(1.0, max(0.0, 1.0 + avg_conf / 5.0))

            duration_s = time.time() - t0

            log.info(f"Transcribed ({duration_s:.1f}s): \"{full_text}\"")
            return TranscriptResult(
                text=full_text,
                audio_hash=audio_hash,
                duration_s=duration_s,
                language=info.language if info else "en",
                confidence=confidence,
            )

        except Exception as exc:
            log.error(f"Transcription failed: {exc}")
            return None
