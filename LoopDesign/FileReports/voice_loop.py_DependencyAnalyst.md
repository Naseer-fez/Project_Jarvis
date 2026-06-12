# Dependency Analysis Report for voice\voice_loop.py

## Library Requirements
- from __future__ import annotations
- from core.voice.stt import SpeechToText
- from core.voice.tts import TextToSpeech
- from core.voice.wake_word import WakeWordDetector
- from typing import Any
- import asyncio
- import inspect
- import logging

## Service Dependencies
- asyncio.get_running_loop

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
