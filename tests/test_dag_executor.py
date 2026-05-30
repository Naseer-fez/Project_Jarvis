import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from core.executor.dag import PlanDAG, DependencyGraphError
from core.executor.engine import DAGExecutor
from core.context.context import TaskExecutionContext
from core.state_machine import StateMachine

class MockObservation:
    def __init__(self, status="success", result=None, error=""):
        self.execution_status = status
        self.result = result or {}
        self.error_message = error
    def to_dict(self):
        return {"status": self.execution_status, "result": self.result, "error": self.error_message}

class MockToolRouter:
    def __init__(self):
        self.executed_steps = []
        self.mock_responses = {}
    async def execute(self, action, params):
        self.executed_steps.append((action, params))
        if action in self.mock_responses:
            res = self.mock_responses[action]
            if isinstance(res, Exception):
                raise res
            return res
        return MockObservation("success", {"message": "ok"})

@pytest.fixture
def mock_context():
    sm = StateMachine()
    ctx = TaskExecutionContext(task_id="test-task-123", trace_id="test-trace-123", state_machine=sm)
    return ctx

@pytest.mark.asyncio
async def test_topological_sort_success():
    steps = [
        {"id": "step_c", "tool": "test_tool", "depends_on": ["step_b"]},
        {"id": "step_a", "tool": "test_tool"},
        {"id": "step_b", "tool": "test_tool", "depends_on": ["step_a"]},
    ]
    dag = PlanDAG(steps)
    sorted_nodes = dag.topological_sort()
    assert sorted_nodes == ["step_a", "step_b", "step_c"]

@pytest.mark.asyncio
async def test_topological_sort_cycle_detected():
    steps = [
        {"id": "step_a", "tool": "test_tool", "depends_on": ["step_b"]},
        {"id": "step_b", "tool": "test_tool", "depends_on": ["step_a"]},
    ]
    dag = PlanDAG(steps)
    with pytest.raises(DependencyGraphError, match="Circular dependency detected"):
        dag.topological_sort()

@pytest.mark.asyncio
async def test_dag_executor_parallel_execution(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a"},
        {"id": "step_b", "tool": "tool_b"},
        {"id": "step_c", "tool": "tool_c", "depends_on": ["step_a", "step_b"]},
    ]
    router = MockToolRouter()
    
    start_times = {}
    end_times = {}
    
    async def slow_execute(action, params):
        start_times[action] = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        end_times[action] = asyncio.get_event_loop().time()
        return MockObservation("success")

    router.execute = slow_execute
    executor = DAGExecutor(tool_router=router)
    
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "success"
    assert abs(start_times["tool_a"] - start_times["tool_b"]) < 0.05
    assert start_times["tool_c"] >= max(end_times["tool_a"], end_times["tool_b"])

@pytest.mark.asyncio
async def test_dag_executor_retry_logic(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a", "retry_count": 2},
    ]
    router = MockToolRouter()
    attempts = 0
    
    async def retry_execute(action, params):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return MockObservation("failed", error="Temporary issue")
        return MockObservation("success", {"data": "finally success"})
        
    router.execute = retry_execute
    executor = DAGExecutor(tool_router=router)
    
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "success"
    assert attempts == 3

@pytest.mark.asyncio
async def test_dag_executor_lifo_rollback(mock_context):
    steps = [
        {
            "id": "step_a",
            "tool": "tool_a",
            "rollback": {"action": "rollback_a", "params": {"key": "val_a"}}
        },
        {
            "id": "step_b",
            "tool": "tool_b",
            "depends_on": ["step_a"],
            "rollback": {"action": "rollback_b", "params": {"key": "val_b"}}
        },
        {
            "id": "step_c",
            "tool": "tool_c",
            "depends_on": ["step_b"]
        }
    ]
    router = MockToolRouter()
    
    router.mock_responses = {
        "tool_a": MockObservation("success"),
        "tool_b": MockObservation("success"),
        "tool_c": MockObservation("failed", error="c failed permanently"),
    }
    
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "failure"
    rollbacks_run = [step for step in router.executed_steps if step[0].startswith("rollback")]
    assert rollbacks_run == [
        ("rollback_b", {"key": "val_b"}),
        ("rollback_a", {"key": "val_a"}),
    ]
