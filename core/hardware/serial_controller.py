"""
core/hardware/serial_controller.py — STUB. Blocked until V3.

This file exists so imports don't break. Every method raises NotImplementedError.
Do not implement until V2 acceptance checklist is fully green.
"""

from __future__ import annotations


class SerialController:
    """Hardware serial controller — blocked until V3."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def send(self, command: str) -> None:
        raise NotImplementedError(
            "SerialController is blocked until V3. "
            "Do not call this in V1 or V2."
        )

    def connect(self, port: str, baud_rate: int = 9600) -> None:
        raise NotImplementedError("SerialController is blocked until V3.")

    def disconnect(self) -> None:
        raise NotImplementedError("SerialController is blocked until V3.")

    @property
    def is_connected(self) -> bool:
        return False
