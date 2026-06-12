# Analysis Report for stt.py

## Dependencies
- __future__.annotations
- asyncio
- logging
- struct
- dataclasses.dataclass
- typing.Any
- typing.Optional

## Schemas
- TranscriptResult
- TranscriptResult attribute: text
- TranscriptResult attribute: audio_hash
- TranscriptResult attribute: duration_s
- TranscriptResult attribute: language
- TranscriptResult attribute: confidence
- STT
- SpeechToText

## API Contracts
- STT.__init__(self, config)
- STT._init(self, config)
- STT.is_ready(self)
- STT._is_speech(self, pcm_bytes, frame_length)
- STT.capture_and_transcribe(self)
- SpeechToText.__init__(self, config)
- SpeechToText._get(self, key, default)
- SpeechToText._choose_backend(self)
- SpeechToText._record_and_transcribe(self)
- SpeechToText._record_and_transcribe_google(self)

## Configuration Variables
- _ENERGY_THRESHOLD

## Assumptions & Notes
- Module Docstring: Speech-to-text for voice mode.

Provides `STT` class with the API expected by V2 acceptance tests:
  - STT._ready (bool attribute)
  - STT.is_ready (property)
  - STT.capture_and_transcribe() -> str | None
  - STT._is_speech(pcm_bytes, frame_length) -> bool
  - STT._vad (porcupine VAD or None)
  - STT._sample_rate (int)
  - TranscriptResult dataclass

Also provides `SpeechToText` (async variant) for the new async controller path.

