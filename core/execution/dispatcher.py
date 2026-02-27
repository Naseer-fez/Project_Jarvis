"""
core/execution/dispatcher.py
-----------------------------
Routes intents to the appropriate handler.
Updated in Phase 5 to support physical_actuate and sensor_read
via the SerialController.
"""

import logging

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Maps classified intents to execution logic.

    Supported intents:
        general_query       -> LLM via Ollama
        physical_actuate    -> SerialController.actuate()
        sensor_read         -> SerialController.read_sensor()
        memory_store        -> HybridMemory store
        memory_recall       -> HybridMemory search
        system_command      -> OS-level action
    """

    def __init__(self, controller, memory, serial_controller=None):
        """
        Args:
            controller:        LLM controller (e.g. OllamaController from Phase 4)
            memory:            HybridMemory instance from Phase 4
            serial_controller: SerialController instance (optional; graceful if None)
        """
        self.controller = controller
        self.memory = memory
        self.serial = serial_controller

        self._handlers = {
            "general_query":    self._handle_general_query,
            "physical_actuate": self._handle_physical_actuate,
            "sensor_read":      self._handle_sensor_read,
            "memory_store":     self._handle_memory_store,
            "memory_recall":    self._handle_memory_recall,
            "system_command":   self._handle_system_command,
        }

    def dispatch(self, intent: str, user_input: str, entities: dict = None) -> str:
        """
        Route intent to handler and return a response string.

        Args:
            intent:     Classified intent label
            user_input: Raw user utterance
            entities:   Extracted entities dict (optional)

        Returns:
            Response string to be spoken/displayed.
        """
        entities = entities or {}
        handler = self._handlers.get(intent, self._handle_general_query)
        logger.info("Dispatching intent='%s' | input='%s'", intent, user_input[:60])

        try:
            return handler(user_input, entities)
        except Exception as e:
            logger.error("Dispatch error for intent '%s': %s", intent, e, exc_info=True)
            return "I encountered an error processing that request."

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _handle_general_query(self, user_input: str, entities: dict) -> str:
        """Pass through to LLM with memory context injection."""
        context = self.memory.get_context(user_input)
        augmented = f"{context}\nUser: {user_input}" if context else user_input
        return self.controller.query(augmented)

    def _handle_physical_actuate(self, user_input: str, entities: dict) -> str:
        """
        Execute a physical device actuation.

        Expects entities:
            device (str): e.g. 'LIGHT', 'FAN'
            state  (str): e.g. 'ON', 'OFF'
        """
        if not self.serial:
            return "Hardware controller is not available."

        device = entities.get("device", "UNKNOWN").upper()
        state = entities.get("state", "TOGGLE").upper()

        result = self.serial.actuate(device, state)

        if result["success"]:
            return f"Done. {device} is now {state}."
        else:
            return f"Failed to actuate {device}. Hardware responded: {result['response']}"

    def _handle_sensor_read(self, user_input: str, entities: dict) -> str:
        """
        Read a sensor value and return a natural language response.

        Expects entities:
            sensor (str): e.g. 'TEMPERATURE', 'HUMIDITY'
        """
        if not self.serial:
            return "Hardware controller is not available."

        sensor = entities.get("sensor", "TEMPERATURE").upper()
        result = self.serial.read_sensor(sensor)

        if result["success"]:
            value = result["value"]
            if sensor == "TEMPERATURE":
                return f"The current temperature is {value} degrees Celsius."
            elif sensor == "HUMIDITY":
                return f"Current humidity is {value} percent."
            else:
                return f"{sensor} reading: {value}."
        else:
            return f"Could not read {sensor} sensor. Check hardware connection."

    def _handle_memory_store(self, user_input: str, entities: dict) -> str:
        content = entities.get("content", user_input)
        self.memory.store(content)
        return "Got it. I've stored that in memory."

    def _handle_memory_recall(self, user_input: str, entities: dict) -> str:
        query = entities.get("query", user_input)
        results = self.memory.search(query)
        if results:
            return f"I found this in memory: {results[0]}"
        return "I don't have anything relevant in memory about that."

    def _handle_system_command(self, user_input: str, entities: dict) -> str:
        """Placeholder for OS-level commands (Phase 6)."""
        return "System commands will be available in the next update."
