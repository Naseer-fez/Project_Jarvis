"""Minimal serial controller with a disabled-by-default safety posture."""

from __future__ import annotations


class SerialController:
    def __init__(
        self,
        config=None,
        port: str | None = None,
        baud_rate: int | None = None,
        timeout: float = 1.0,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self._serial = None
        self.enabled = False
        self.default_port = port
        self.baud_rate = int(baud_rate or 115200)

        if config is not None:
            try:
                self.enabled = config.getboolean("hardware", "enabled", fallback=False)
            except Exception:
                self.enabled = False
            try:
                self.default_port = self.default_port or config.get(
                    "hardware",
                    "default_port",
                    fallback=None,
                )
            except Exception:
                pass
            try:
                self.baud_rate = int(
                    config.get("hardware", "baud_rate", fallback=str(self.baud_rate))
                )
            except Exception:
                pass

        if self.enabled and self.default_port:
            self.connect(self.default_port)

    @property
    def is_connected(self) -> bool:
        return bool(self._serial is not None and getattr(self._serial, "is_open", False))

    def connect(self, port: str | None = None):
        if not self.enabled:
            raise NotImplementedError("Hardware serial control is disabled by config.")

        target_port = port or self.default_port
        if not target_port:
            raise ValueError("No serial port configured.")

        import serial

        self._serial = serial.Serial(
            target_port,
            baudrate=self.baud_rate,
            timeout=self.timeout,
        )
        self.default_port = target_port
        return self

    def send(self, command: str):
        if not self.enabled:
            raise NotImplementedError("Hardware serial control is disabled by config.")
        if not self.is_connected:
            raise RuntimeError("Serial controller is not connected.")

        payload = f"{command}\n".encode("utf-8")
        self._serial.write(payload)
        if hasattr(self._serial, "flush"):
            self._serial.flush()
        if hasattr(self._serial, "readline"):
            return self._serial.readline().decode("utf-8", errors="replace").strip()
        return "OK"

    def close(self) -> None:
        if self._serial is not None and hasattr(self._serial, "close"):
            self._serial.close()


__all__ = ["SerialController"]
