# Dependency Analysis Report for controller\web_search.py

## Library Requirements
- from __future__ import annotations
- from core.controller.request_rules import is_explicit_web_search
- from core.controller.request_rules import is_preference_relevant
- from core.controller.request_rules import should_force_web_search
- from core.tools.web_tools import _basic_query_cleanup
- from core.tools.web_tools import web_search
- from typing import Any
- import logging
- import re

## Service Dependencies
- None detected

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
