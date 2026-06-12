# Dependency Analysis Report for context\context.py

## Library Requirements
- from __future__ import annotations
- from contextvars import Token
- from core.logging.logger import reset_trace_ids
- from core.logging.logger import set_trace_ids
- from core.state_machine import State
- from core.state_machine import StateMachine
- from pathlib import Path
- from typing import Any
- import asyncio
- import json
- import logging
- import uuid

## Service Dependencies
- asyncio.to_thread

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
