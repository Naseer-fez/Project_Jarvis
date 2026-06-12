# API Analyst Report: voice\stt.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import struct`
- `from dataclasses import dataclass`
- `from typing import Any`
- `from typing import Optional`

## Configuration Variables
- `_ENERGY_THRESHOLD` = `500`
- `_VAD_ENERGY_THRESHOLD` = `_ENERGY_THRESHOLD`

## Schemas & API Contracts (Classes)

### Class `TranscriptResult`
> Structured output from speech recognition.

**Fields/Schema:**
  - `text: str`
  - `audio_hash: str`
  - `duration_s: float`
  - `language: str`
  - `confidence: float`



### Class `STT`
> Synchronous STT wrapper with graceful degradation.

Attributes exposed for tests:
  _ready         — True once the backend model is loaded
  _vad           — VAD object or None when using energy-based detection
  _sample_rate   — audio sample rate in Hz

**Methods:**
- `def __init__(self, config: Any=None) -> None`
- `def _init(self, config: Any) -> None`
  - *Attempt to load the whisper model. Sets _ready on success.*
- @property
- `def is_ready(self) -> bool`
- `def _is_speech(self, pcm_bytes: bytes, frame_length: int) -> bool`
  - *Return True if the PCM frame contains speech.*
- `def capture_and_transcribe(self) -> Optional[str]`
  - *Record audio and return the transcribed text (or None if not ready).*


### Class `SpeechToText`
> Async STT wrapper with graceful fallback behavior.

**Methods:**
- `def __init__(self, config: Any) -> None`
- `def _get(self, key: str, default: str) -> str`
- `def _choose_backend(self) -> str`
- `async def transcribe(self) -> str`
- `async def _read_from_input(self) -> str`
- `def _record_and_transcribe(self) -> str`
- `def _record_and_transcribe_google(self) -> str`

