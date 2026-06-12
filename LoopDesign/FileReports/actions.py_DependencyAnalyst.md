# Dependency Analysis Report for desktop\actions.py

## Library Requirements
- from __future__ import annotations
- from core.autonomy.risk_evaluator import RiskEvaluator
- from core.autonomy.risk_evaluator import RiskLevel
- from core.autonomy.risk_evaluator import RiskResult
- from core.desktop.contracts import DesktopAction
- from core.desktop.contracts import DesktopActionResult
- from core.desktop.contracts import DesktopActionStatus
- from core.desktop.contracts import DesktopActionType
- from core.desktop.contracts import DesktopRiskTier
- from core.logging.logger import audit
- from core.tools.gui_control import click
- from core.tools.gui_control import click_screen_target
- from core.tools.gui_control import click_text_on_screen
- from core.tools.gui_control import clipboard_get
- from core.tools.gui_control import clipboard_paste
- from core.tools.gui_control import clipboard_set
- from core.tools.gui_control import double_click
- from core.tools.gui_control import double_click_screen_target
- from core.tools.gui_control import drag
- from core.tools.gui_control import focus_window
- from core.tools.gui_control import hotkey
- from core.tools.gui_control import move_mouse
- from core.tools.gui_control import press_key
- from core.tools.gui_control import right_click
- from core.tools.gui_control import right_click_screen_target
- from core.tools.gui_control import scroll
- from core.tools.gui_control import type_text
- from core.tools.system_automation import async_launch_application
- from typing import Any
- from typing import Callable
- from typing import cast
- import inspect
- import time

## Service Dependencies
- None detected

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
