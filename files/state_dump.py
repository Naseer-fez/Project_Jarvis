"""
core/introspection/state_dump.py

Full agent state snapshot for debugging, auditing, and health dashboards.
Aggregates data from all subsystems into a single serialisable dict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.agentic.agent_context import AgentContext
    from core.agentic.goal_manager import GoalManager
    from core.agentic.belief_state import BeliefState
    from core.agentic.decision_trace import DecisionTrace
    from core.agentic.scheduler import Scheduler
    from core.autonomy.override import HumanOverrideProtocol


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def dump_state(
    context: Optional[Any] = None,
    goal_manager: Optional[Any] = None,
    belief_state: Optional[Any] = None,
    decision_trace: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    override_protocol: Optional[Any] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Collect a full snapshot of agent state from all available subsystems.

    Pass whichever subsystems are available; omitted ones are skipped.

    Returns:
        dict with keys: "captured_at", "context", "goals", "beliefs",
                        "trace", "schedule", "overrides", and any "extra".
    """
    snapshot: dict[str, Any] = {"captured_at": _utcnow().isoformat()}

    if context is not None:
        snapshot["context"] = context.snapshot()

    if goal_manager is not None:
        snapshot["goals"] = goal_manager.snapshot()

    if belief_state is not None:
        snapshot["beliefs"] = belief_state.snapshot()

    if decision_trace is not None:
        snapshot["trace"] = decision_trace.snapshot()

    if scheduler is not None:
        snapshot["schedule"] = scheduler.snapshot()

    if override_protocol is not None:
        snapshot["overrides"] = override_protocol.snapshot()

    if extra:
        snapshot["extra"] = extra

    return snapshot


def pretty_dump(
    context=None, goal_manager=None, belief_state=None,
    decision_trace=None, scheduler=None, override_protocol=None,
) -> str:
    """Return a pretty-printed JSON string of the full state dump."""
    import json
    state = dump_state(
        context=context,
        goal_manager=goal_manager,
        belief_state=belief_state,
        decision_trace=decision_trace,
        scheduler=scheduler,
        override_protocol=override_protocol,
    )
    return json.dumps(state, indent=2, default=str)
