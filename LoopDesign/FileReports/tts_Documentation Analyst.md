# Analysis Report for tts.py

## Dependencies
- __future__.annotations
- logging
- re
- threading
- typing.Any

## Schemas
- TTS

## API Contracts
- _split_sentences(text)
- TTS.__init__(self, config)
- TTS._init_backend(self, config)
- TTS.is_speaking(self)
- TTS.speak(self, text)
- TTS.stop(self)
- TTS._speak_sentence(self, sentence)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Text-to-speech with fallback chain: edge-tts -> pyttsx3 -> print-to-stdout.

The TTS class provides the synchronous API expected by the V2 acceptance tests:
  - TTS(config)
  - tts.speak(text)          — synchronous
  - tts.stop()               — interrupt ongoing speech
  - tts.is_speaking          — bool property
  - tts._backend             — "edge", "pyttsx3", or "cli"
  - tts._stop_event          — threading.Event
  - tts._init_backend(...)   — callable, returns backend name (patchable in tests)

The TextToSpeech class is the async variant kept for backward compat.

