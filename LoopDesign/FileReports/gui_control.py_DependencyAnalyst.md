# Dependency Analysis Report for tools\gui_control.py

## Library Requirements
- from __future__ import annotations
- from core.tools.screen import capture_screen
- from core.tools.screen import find_text_on_screen
- from core.tools.vision import VisionTool
- from core.types.common import ToolResult
- from pathlib import Path
- from typing import Any
- import asyncio
- import configparser
- import json
- import logging
- import pyautogui
- import pygetwindow
- import pyperclip
- import re
- import time

## Service Dependencies
- asyncio.sleep
- asyncio.to_thread

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
