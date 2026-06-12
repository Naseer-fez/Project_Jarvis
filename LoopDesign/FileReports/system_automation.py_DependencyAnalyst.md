# Dependency Analysis Report for tools\system_automation.py

## Library Requirements
- from core.tools.path_utils import _assert_safe_path
- from dataclasses import dataclass
- from dataclasses import field
- from pathlib import Path
- import asyncio
- import logging
- import os
- import shlex
- import subprocess

## Service Dependencies
- asyncio.create_subprocess_exec
- asyncio.to_thread
- asyncio.wait_for

## Hidden Execution Links
- subprocess.Popen

## Configurations / Assumptions
- Analyzed via AST and regex.
