# API Analyst Report: tools\auto_clicker.py

## Dependencies
- `import argparse`
- `import asyncio`
- `import logging`
- `import sys`
- `from core.tools.gui_control import click_screen_target`

## Functions & Endpoints

### `run_auto_clicker`
`async def run_auto_clicker(target: str, interval: float, continuous: bool, min_confidence: float) -> None`
> Run the auto clicker loop.

### `main`
`def main() -> None`