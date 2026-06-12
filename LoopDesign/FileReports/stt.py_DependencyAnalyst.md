# Dependency Analysis Report for voice\stt.py

## Library Requirements
- from __future__ import annotations
- from dataclasses import dataclass
- from faster_whisper import WhisperModel
- from typing import Any
- from typing import Optional
- import asyncio
- import logging
- import numpy
- import sounddevice
- import speech_recognition
- import struct

## Service Dependencies
- asyncio.get_running_loop

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
