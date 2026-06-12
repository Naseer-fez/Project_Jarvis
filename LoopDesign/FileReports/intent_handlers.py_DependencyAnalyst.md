# Dependency Analysis Report for controller\intent_handlers.py

## Library Requirements
- from core.controller.request_rules import is_active_window_request
- from core.controller.request_rules import is_explicit_web_search
- from core.controller.web_search import handle_web_search
- from core.controller_v2 import JarvisControllerV2
- from core.desktop.shortcuts import handle_desktop_command
- from core.desktop.shortcuts import plan_desktop_command
- from typing import TYPE_CHECKING
- import logging
- import uuid

## Service Dependencies
- None detected

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
