"""
core/kernel/entrypoint.py
─────────────────────────
Lifecycle entrypoint exporting async_run startup and shutdown routines.
"""

from __future__ import annotations

from core.runtime.entrypoint import async_run

__all__ = ["async_run"]
