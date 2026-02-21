"""
core/hardware/serial_controller.py - Optional serial bridge for V3 hardware actions.

Safe default: disabled unless explicitly enabled in config or constructor.
"""

from __future__ import annotations

from typing import Any


class SerialController:
    def __init__(self, config=None, enabled: bool | None = None) -> None:
        self._serial = None
        self._port: str | None = None
        self._baud_rate: int = 9600

        config_enabled = False
        if config is not None:
            try:
                config_enabled = config.getboolean("hardware", "enabled", fallback=False)
            except Exception:
                config_enabled = False

        self._enabled = config_enabled if enabled is None else bool(enabled)

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise NotImplementedError(
                "SerialController is disabled. Set [hardware] enabled=true to unblock V3 serial commands."
            )

    def connect(self, port: str, baud_rate: int = 9600, timeout_s: float = 1.0) -> None:
        self._require_enabled()
        if not port:
            raise ValueError("Serial port is required.")

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for SerialController.") from exc

        if self._serial and self._serial.is_open:
            self.disconnect()

        self._serial = serial.Serial(port=port, baudrate=int(baud_rate), timeout=timeout_s)
        self._port = port
        self._baud_rate = int(baud_rate)

    def send(self, command: str) -> str:
        self._require_enabled()
        if not command:
            raise ValueError("Serial command cannot be empty.")
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Serial port is not connected.")

        payload = (command.strip() + "\n").encode("utf-8")
        self._serial.write(payload)
        self._serial.flush()

        try:
            reply = self._serial.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            reply = ""
        return reply or "ok"

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None
        self._port = None

    @property
    def is_connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    @property
    def enabled(self) -> bool:
        return self._enabled
