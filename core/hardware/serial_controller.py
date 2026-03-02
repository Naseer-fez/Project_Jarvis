"""
core/hardware/serial_controller.py
-----------------------------------
Manages serial communication with external hardware (e.g., Arduino).

Protocol (simple text-based):
  JARVIS -> Arduino: "CMD:<command>:<value>\n"
  Arduino -> JARVIS: "ACK:<command>:<result>\n"  |  "ERR:<message>\n"

Falls back to simulation mode gracefully if hardware is absent.
"""

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)

CMD_PREFIX = "CMD"
ACK_PREFIX = "ACK"
ERR_PREFIX = "ERR"
SENSOR_READ_CMD = "SENSOR_READ"
ACTUATE_CMD = "ACTUATE"


class SerialController:
    """
    Thread-safe serial controller with simulation fallback.

    In simulation mode, commands are logged but not sent to hardware.
    This ensures the JARVIS loop never crashes due to missing hardware.
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict, ConfigParser, or None.
                When dict: keys are 'com_port', 'baud_rate', 'timeout_seconds', 'require_hardware'.
                When ConfigParser: reads from [hardware] or [serial] section.
                When None: defaults are used (simulation mode).
        """
        import configparser as _configparser

        if config is None:
            cfg_dict: dict = {}
        elif isinstance(config, _configparser.ConfigParser):
            # Try [hardware] then [serial], fall back to empty
            section = None
            for sec in ("hardware", "serial", "execution"):
                if config.has_section(sec):
                    section = sec
                    break
            if section:
                cfg_dict = dict(config[section])
            else:
                cfg_dict = {}
        else:
            cfg_dict = config  # assume dict-like

        self.port = cfg_dict.get("com_port", "COM7")
        self.baud_rate = int(cfg_dict.get("baud_rate", 115200))
        self.timeout = float(cfg_dict.get("timeout_seconds", 2))
        require_hw_raw = cfg_dict.get("require_hardware", "false")
        self.require_hardware = (
            require_hw_raw.lower() == "true" if isinstance(require_hw_raw, str) else bool(require_hw_raw)
        )

        self._serial = None
        self._lock = threading.Lock()
        self._simulation_mode = False
        self._connected = False

        self._connect()

    def _connect(self):
        """Attempt to open the serial port. Falls back to simulation if unavailable."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
            )
            # Give Arduino time to reset after serial connection
            time.sleep(2.0)
            self._connected = True
            self._simulation_mode = False
            logger.info(
                "Serial connected | port=%s baud=%d", self.port, self.baud_rate
            )

        except Exception as e:
            if self.require_hardware:
                raise RuntimeError(
                    f"Hardware required but serial connection failed on {self.port}: {e}"
                ) from e

            logger.warning(
                "Serial port %s unavailable (%s). "
                "Running in SIMULATION mode — hardware commands will be logged only.",
                self.port, e,
            )
            self._simulation_mode = True
            self._connected = False

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return self._connected and not self._simulation_mode

    # ── Legacy stub API (used by V1/V2 acceptance tests) ─────────────────

    def send(self, command: str) -> None:
        """Legacy stub — raises NotImplementedError (serial controller unimplemented)."""
        raise NotImplementedError(
            "SerialController.send() is not implemented. "
            "Use send_command() for real hardware or simulation."
        )

    def connect(self, port: str | None = None, baud_rate: int | None = None) -> None:
        """Legacy stub — raises NotImplementedError."""
        raise NotImplementedError(
            f"SerialController.connect({port!r}) is not implemented. "
            "The controller auto-connects in __init__."
        )

    def send_command(self, command: str, value: str = "") -> str:
        """
        Send a command to the hardware and return the response.

        Args:
            command: Command name (e.g., 'LIGHT', 'FAN')
            value:   Command value (e.g., 'ON', 'OFF', 'READ')

        Returns:
            Response string from device, or simulated response.
        """
        msg = f"{CMD_PREFIX}:{command}:{value}\n"

        if self._simulation_mode:
            simulated = f"{ACK_PREFIX}:{command}:SIMULATED"
            logger.info("[SIM] TX: %s | RX: %s", msg.strip(), simulated)
            return simulated

        with self._lock:
            try:
                self._serial.reset_input_buffer()
                self._serial.write(msg.encode("utf-8"))
                self._serial.flush()

                response = self._serial.readline().decode("utf-8").strip()
                logger.debug("Serial TX: %s | RX: %s", msg.strip(), response)

                if not response:
                    return f"{ERR_PREFIX}:TIMEOUT"
                return response

            except Exception as e:
                logger.error("Serial communication error: %s", e)
                return f"{ERR_PREFIX}:{e}"

    def actuate(self, device: str, state: str) -> dict:
        """
        Send an actuation command (e.g., turn light on/off).

        Args:
            device: Target device name (e.g., 'LIGHT', 'FAN', 'LOCK')
            state:  Desired state (e.g., 'ON', 'OFF', 'TOGGLE')

        Returns:
            dict with 'success' (bool) and 'response' (str)
        """
        logger.info("Actuate: %s -> %s", device, state)
        response = self.send_command(ACTUATE_CMD, f"{device}:{state}")
        success = response.startswith(ACK_PREFIX)
        return {"success": success, "response": response, "device": device, "state": state}

    def read_sensor(self, sensor: str) -> dict:
        """
        Request a sensor reading.

        Args:
            sensor: Sensor identifier (e.g., 'TEMPERATURE', 'HUMIDITY', 'MOTION')

        Returns:
            dict with 'success' (bool), 'sensor', and 'value' (str)
        """
        logger.info("Sensor read: %s", sensor)
        response = self.send_command(SENSOR_READ_CMD, sensor)

        if response.startswith(ACK_PREFIX):
            parts = response.split(":")
            value = parts[2] if len(parts) > 2 else "UNKNOWN"
            return {"success": True, "sensor": sensor, "value": value, "raw": response}

        return {"success": False, "sensor": sensor, "value": None, "raw": response}

    def close(self):
        """Clean up serial connection."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
                logger.info("Serial port %s closed.", self.port)
            except Exception as e:
                logger.warning("Error closing serial port: %s", e)
        self._connected = False

    # ── Async extensions (Session 7) ────────────────────────────────────────

    async def async_send_command(self, cmd: str, value: str = "") -> dict:
        """Async wrapper around send_command; offloads blocking I/O to executor."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.send_command, cmd, value)
        return {"success": True, "response": str(result), "simulated": self._simulation_mode}

    async def firmware_ping(self) -> bool:
        """Return True if the device responds with PONG to a PING command."""
        if self._simulation_mode:
            return True
        try:
            r = await self.async_send_command("PING")
            return "PONG" in str(r.get("response", ""))
        except Exception:
            return False

    async def sensor_read_loop(self, callback, interval: float = 1.0) -> None:
        """Continuously poll sensors and invoke callback with each reading."""
        import random
        while getattr(self, "_running", True):
            await asyncio.sleep(interval)
            if self._simulation_mode:
                data = {
                    "temperature": round(random.uniform(20.0, 25.0), 1),
                    "humidity": round(random.uniform(40.0, 60.0), 1),
                    "simulated": True,
                }
            else:
                data = await self.async_send_command("READ_SENSORS")
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
