# Dependency Analysis Report for runtime\bootstrap.py

## Library Requirements
- from __future__ import annotations
- from core.config import load_config
- from core.controller_v2 import JarvisControllerV2
- from core.introspection.health import HealthCheck
- from core.introspection.health import HealthReport
- from core.introspection.health import HealthStatus
- from core.introspection.health import run_startup_health_check
- from core.llm.model_router import ModelRouter
- from core.logging import logger
- from core.ops.production import validate_production_config
- from core.runtime.paths import PROJECT_ROOT
- from core.runtime.paths import _resolve_path
- from dotenv import load_dotenv
- from integrations.loader import IntegrationLoader
- from integrations.registry import integration_registry
- from pathlib import Path
- from typing import Any
- import argparse
- import asyncio
- import configparser
- import contextlib
- import dataclasses
- import faulthandler
- import io
- import json
- import logging
- import math
- import os
- import signal
- import sys
- import threading

## Service Dependencies
- asyncio.Event

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
