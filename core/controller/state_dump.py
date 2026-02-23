# core/controller/state_dump.py
# Serializes / dumps internal controller state for debug, recovery, and audit.

from typing import Any


def dump_state(state: Any) -> dict:
    """
    Serialize the given controller state into a plain dict.

    Args:
        state: Any object or dict representing internal state.

    Returns:
        A dict representation of the state.
    """
    try:
        if isinstance(state, dict):
            return dict(state)
        elif hasattr(state, "__dict__"):
            return vars(state).copy()
        else:
            return {"raw": str(state)}
    except Exception as e:
        return {"error": str(e), "raw": None}


class StateDumper:
    """Helper class for incremental state capture."""

    def __init__(self):
        self._snapshots: list = []

    def capture(self, state: Any) -> None:
        """Capture a snapshot of the current state."""
        self._snapshots.append(dump_state(state))

    def get_snapshots(self) -> list:
        """Return all captured snapshots."""
        return list(self._snapshots)

    def clear(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
