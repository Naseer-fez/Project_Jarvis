"""
core/agent/state_machine.py
Shim that re-exports StateMachine and AgentState from
the canonical core.state_machine module for import compatibility.
"""
from core.state_machine import StateMachine, State as AgentState

__all__ = ["StateMachine", "AgentState"]
