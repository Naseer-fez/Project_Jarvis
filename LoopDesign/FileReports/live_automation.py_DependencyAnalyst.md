# Dependency Analysis Report for automation\live_automation.py

## Library Requirements
- from __future__ import annotations
- from core.automation.payload_extractor import PayloadExtractor
- from core.automation.rag_ingester import RagIngester
- from core.automation.scan_pipeline import ScanBatch
- from core.automation.scan_pipeline import ScanPipeline
- from core.automation.scan_rules import ScanRoute
- from core.automation.scan_rules import build_scan_routes
- from core.context.context import TaskExecutionContext
- from core.runtime.paths import _resolve_path
- from dataclasses import asdict
- from dataclasses import dataclass
- from datetime import datetime
- from datetime import timezone
- from pathlib import Path
- from typing import Any
- from typing import Awaitable
- from typing import Callable
- import asyncio
- import contextlib
- import hashlib
- import json
- import logging
- import re
- import shutil
- import time

## Service Dependencies
- asyncio.create_task
- asyncio.get_running_loop
- asyncio.sleep
- asyncio.to_thread
- shutil.move

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
