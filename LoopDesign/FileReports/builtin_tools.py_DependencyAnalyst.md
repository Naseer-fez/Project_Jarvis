# Dependency Analysis Report for tools\builtin_tools.py

## Library Requirements
- from core.tools.fast_search_tool import run_fast_search
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
- from core.tools.gui_control import get_active_window
- from core.tools.gui_control import hotkey
- from core.tools.gui_control import move_mouse
- from core.tools.gui_control import press_key
- from core.tools.gui_control import right_click
- from core.tools.gui_control import right_click_screen_target
- from core.tools.gui_control import scroll
- from core.tools.gui_control import type_text
- from core.tools.hardware_tools import list_hardware_devices
- from core.tools.hardware_tools import ping_device
- from core.tools.hardware_tools import read_sensor
- from core.tools.hardware_tools import send_hardware_command
- from core.tools.path_utils import ALLOWED_DIRECTORIES
- from core.tools.path_utils import _PROJECT_ROOT
- from core.tools.path_utils import _assert_safe_path
- from core.tools.screen import capture_region
- from core.tools.screen import capture_screen
- from core.tools.screen import describe_screen
- from core.tools.screen import find_text_on_screen
- from core.tools.screen import read_screen_text
- from core.tools.screen import wait_for_text_on_screen
- from core.tools.system_automation import async_delete_file
- from core.tools.system_automation import async_execute_shell
- from core.tools.system_automation import async_launch_application
- from core.tools.system_automation import async_write_file
- from core.tools.universal_converter import perform_conversion
- from core.tools.web_tools import configure_web_tools
- from core.tools.web_tools import web_scrape
- from core.tools.web_tools import web_search
- from pathlib import Path
- import asyncio
- import datetime
- import fnmatch
- import json
- import logging
- import os
- import platform
- import psutil
- import shutil

## Service Dependencies
- asyncio.to_thread
- shutil.copy2
- shutil.move

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
