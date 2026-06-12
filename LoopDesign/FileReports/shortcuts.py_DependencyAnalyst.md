# Dependency Analysis Report for desktop\shortcuts.py

## Library Requirements
- from __future__ import annotations
- from core.desktop.actions import DesktopActionExecutor
- from core.desktop.contracts import DesktopAction
- from core.desktop.contracts import DesktopActionType
- from core.desktop.mission import DesktopMissionExecutor
- from core.desktop.mission import DesktopMissionStatus
- from core.desktop.observation import DesktopObserver
- from core.tools.system_automation import async_launch_application
- from dataclasses import dataclass
- from pathlib import Path
- from urllib.parse import quote_plus
- import re

## Service Dependencies
- URL: https://www.bing.com/search?q={quote_plus(query)}

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
