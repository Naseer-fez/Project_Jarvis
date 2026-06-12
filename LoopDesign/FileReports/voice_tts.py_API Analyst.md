# API Analyst Report: voice\tts.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `import re`
- `import threading`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `TTS`
> Synchronous text-to-speech with a three-tier fallback chain.

Backend priority: edge-tts  →  pyttsx3  →  CLI (print to stdout)
All backend errors are caught silently; the next fallback is tried.

Design decisions:
  * _init_backend() is a separate method so tests can monkeypatch it.
  * _stop_event is a threading.Event that halts sentence-by-sentence output.
  * speak() is blocking (runs in the calling thread) so tests can assert on
    stdout without races.

**Methods:**
- `def __init__(self, config: Any) -> None`
- `def _init_backend(self, config: Any) -> str`
  - *Detect the best available backend based on configured priority.*
- @property
- `def is_speaking(self) -> bool`
- `def speak(self, text: str) -> None`
  - *Speak *text* synchronously, respecting stop_event between sentences.*
- `def stop(self) -> None`
  - *Request interruption of ongoing speech.*
- `def _speak_sentence(self, sentence: str) -> None`
  - *Speak a single sentence using the configured backend.*


## Functions & Endpoints

### `_split_sentences`
`def _split_sentences(text: str) -> list[str]`
> Split text into sentences for streaming TTS output.
