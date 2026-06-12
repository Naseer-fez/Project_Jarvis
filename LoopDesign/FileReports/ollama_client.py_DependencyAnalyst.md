# Dependency Analysis Report for llm\ollama_client.py

## Library Requirements
- from __future__ import annotations
- from core.config.defaults import OLLAMA_BASE_URL
- from core.llm.defaults import DEFAULT_MODEL
- from typing import Any
- from urllib.request import urlopen
- import aiohttp
- import asyncio
- import json
- import logging
- import re

## Service Dependencies
- aiohttp.ClientSession
- aiohttp.ClientTimeout
- asyncio.sleep

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
