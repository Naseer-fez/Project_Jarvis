# Dependency Analysis Report for voice\wake_word.py

## Library Requirements
- from __future__ import annotations
- from pvrecorder import PvRecorder
- from typing import Any
- from typing import Callable
- from typing import Optional
- import asyncio
- import logging
- import os
- import pvporcupine
- import threading

## Service Dependencies
- URL: https://console.picovoice.ai/
- asyncio.get_running_loop
- asyncio.sleep

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
