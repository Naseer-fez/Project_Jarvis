# Dependency Analysis Report for tools\web_tools.py

## Library Requirements
- from __future__ import annotations
- from bs4 import BeautifulSoup
- from dataclasses import dataclass
- from duckduckgo_search import DDGS
- from pathlib import Path
- from typing import Any
- from typing import Protocol
- from typing import cast
- import asyncio
- import configparser
- import json
- import logging
- import os
- import re
- import requests

## Service Dependencies
- URL: https://api.tavily.com/search
- asyncio.to_thread
- asyncio.wait_for
- requests.get
- requests.post

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
