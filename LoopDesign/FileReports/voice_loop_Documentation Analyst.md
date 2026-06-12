# Analysis Report for voice_loop.py

## Dependencies
- __future__.annotations
- asyncio
- inspect
- logging
- typing.Any
- core.voice.stt.SpeechToText
- core.voice.tts.TextToSpeech
- core.voice.wake_word.WakeWordDetector

## Schemas
- VoiceLoop

## API Contracts
- VoiceLoop.__init__(self, controller, config)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Voice loop orchestration: wake -> transcribe -> process -> speak.

