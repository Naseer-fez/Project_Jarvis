from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agent.agent_loop import AgentLoopEngine
from core.context.context import TaskExecutionContext
from core.state_machine import State, StateMachine
from core.runtime.container import ServiceContainer
from core.executor.engine import DAGExecutor


class FailedObservation:
    def __init__(self, error_message: str = "tool failed") -> None:
        self.tool_name = "dummy_tool"
        self.arguments: dict[str, object] = {}
        self.execution_status = "failure"
        self.output_summary = ""
        self.error_message = error_message
        self.duration_seconds = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "execution_status": self.execution_status,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }


class FailingToolRouter:
    async def execute(self, action, params):
        return FailedObservation("simulated tool failure")


@pytest.mark.asyncio
async def test_agent_loop_failure_stops_cleanly_without_invalid_transition():
    planner = MagicMock()
    planner.plan = AsyncMock(
        return_value={
            "steps": [
                {"id": 1, "tool": "dummy_tool", "params": {}},
            ]
        }
    )

    risk_result = type(
        "RiskResult",
        (),
        {
            "level": type("Level", (), {"label": staticmethod(lambda: "low")})(),
            "blocking_actions": set(),
            "confirm_actions": set(),
            "high_risk_actions": set(),
            "is_blocked": False,
            "requires_confirmation": False,
        },
    )()

    risk_evaluator = MagicMock()
    risk_evaluator.evaluate_plan.return_value = risk_result

    autonomy_governor = MagicMock()
    autonomy_governor.can_execute.return_value = (True, "")

    context = TaskExecutionContext(
        task_id="task-failure",
        trace_id="trace-failure",
        state_machine=StateMachine(),
    )
    container = ServiceContainer()
    container.register("dag_executor", DAGExecutor, is_singleton=False)
    container.register("state_machine", StateMachine, is_singleton=False)

    engine = AgentLoopEngine(
        task_planner=planner,
        tool_router=FailingToolRouter(),  # type: ignore[arg-type]
        risk_evaluator=risk_evaluator,
        autonomy_governor=autonomy_governor,
        llm=MagicMock(),
        container=container,
    )

    trace = await engine.run("fail on purpose", context)

    assert trace.success is False
    assert trace.stop_reason == "unrecoverable_tool_failure"
    assert trace.final_response == "Task execution failed."
    assert context.state_machine.state == State.IDLE
