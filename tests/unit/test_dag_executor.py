import pytest
import asyncio
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
            import inspect
            if inspect.iscoroutinefunction(res) or callable(res):
                return await res() if inspect.iscoroutinefunction(res) else res()
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
    """
    Verify parallel execution of independent nodes without wall-clock sleep.
    Uses asyncio.Event to enforce an execution order that proves concurrency.
    """
    steps = [
        {"id": "step_a", "tool": "tool_a"},
        {"id": "step_b", "tool": "tool_b"},
        {"id": "step_c", "tool": "tool_c", "depends_on": ["step_a", "step_b"]},
    ]
    
    router = MockToolRouter()
    
    a_started = asyncio.Event()
    b_started = asyncio.Event()

    async def execute_a(action, params):
        a_started.set()
        await b_started.wait()  # Wait for B to start to prove concurrency
        return MockObservation("success")

    async def execute_b(action, params):
        b_started.set()
        await a_started.wait()  # Wait for A to start to prove concurrency
        return MockObservation("success")

    async def execute_c(action, params):
        # By the time C runs, A and B must both be finished (events set)
        assert a_started.is_set()
        assert b_started.is_set()
        return MockObservation("success")
        
    async def custom_execute(action, params):
        if action == "tool_a":
            return await execute_a(action, params)
        elif action == "tool_b":
            return await execute_b(action, params)
        elif action == "tool_c":
            return await execute_c(action, params)

    router.execute = custom_execute
    executor = DAGExecutor(tool_router=router)
    
    plan = {"steps": steps}
    
    # Run the executor; it will hang if A and B don't run concurrently.
    result = await asyncio.wait_for(executor.execute(plan, mock_context), timeout=2.0)
    
    assert result["status"] == "success"


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
    # Assert LIFO order
    assert rollbacks_run == [
        ("rollback_b", {"key": "val_b"}),
        ("rollback_a", {"key": "val_a"}),
    ]

@pytest.mark.asyncio
async def test_dag_executor_rollback_continues_on_failure(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a", "rollback": {"action": "rollback_a"}},
        {"id": "step_b", "tool": "tool_b", "depends_on": ["step_a"], "rollback": {"action": "rollback_b"}},
        {"id": "step_c", "tool": "tool_c", "depends_on": ["step_b"]}
    ]
    router = MockToolRouter()
    
    router.mock_responses = {
        "tool_a": MockObservation("success"),
        "tool_b": MockObservation("success"),
        "tool_c": MockObservation("failed", error="c failed permanently"),
        "rollback_b": Exception("Rollback B failed!"),
        "rollback_a": MockObservation("success")
    }
    
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "failure"
    rollbacks_run = [step for step in router.executed_steps if step[0].startswith("rollback")]
    assert rollbacks_run == [
        ("rollback_b", {}),
        ("rollback_a", {})
    ]

@pytest.mark.asyncio
async def test_dag_executor_missing_dependency(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a", "depends_on": ["missing_step"]},
    ]
    router = MockToolRouter()
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "success"
    assert router.executed_steps == [("tool_a", {})]

@pytest.mark.asyncio
async def test_dag_executor_autonomy_governor_blocked(mock_context):
    class MockGovernor:
        def can_execute(self, action):
            if action == "forbidden_tool":
                return False, "Not allowed"
            return True, ""
            
    steps = [
        {"id": "step_a", "tool": "forbidden_tool"}
    ]
    router = MockToolRouter()
    executor = DAGExecutor(tool_router=router, autonomy_governor=MockGovernor())
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "failure"
    assert result["failed_steps"] == ["step_a"]
    assert "Autonomy Governor blocked" in result["results"]["step_a"]["error"]

@pytest.mark.asyncio
async def test_dag_executor_unhandled_exception(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a"}
    ]
    router = MockToolRouter()
    router.mock_responses = {
        "tool_a": ValueError("Random unexpected error")
    }
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    result = await executor.execute(plan, mock_context)
    
    assert result["status"] == "failure"
    assert result["failed_steps"] == ["step_a"]
    assert "Random unexpected error" in result["results"]["step_a"]["error"]


@pytest.mark.asyncio
async def test_dag_executor_cancellation(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a"}
    ]
    router = MockToolRouter()
    
    async def cancel_execute(action, params):
        raise asyncio.CancelledError("Simulated task cancellation")
        
    router.execute = cancel_execute
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    
    result = await executor.execute(plan, mock_context)
    assert "Cancelled" in result["results"]["step_a"]["error"]



class CustomBaseException(BaseException):
    pass

@pytest.mark.asyncio
async def test_dag_executor_base_exception(mock_context):
    steps = [
        {"id": "step_a", "tool": "tool_a"}
    ]
    router = MockToolRouter()
    
    async def crash_execute(action, params):
        raise CustomBaseException("Simulated critical error")
        
    router.execute = crash_execute
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    
    result = await executor.execute(plan, mock_context)
    assert result["status"] == "failure"
    assert "step_a" in result["failed_steps"]
    assert "BaseException: CustomBaseException" in result["results"]["step_a"]["error"]


@pytest.mark.asyncio
async def test_dag_executor_concurrency_race(mock_context):
    """
    Test that condition variables properly synchronize DAG completion status 
    even when many concurrent tasks finish at the exact same moment.
    """
    steps = [{"id": f"step_{i}", "tool": f"tool_{i}"} for i in range(10)]
    steps.append({"id": "step_final", "tool": "tool_final", "depends_on": [f"step_{i}" for i in range(10)]})
    
    router = MockToolRouter()
    sync_event = asyncio.Event()

    async def sync_execute(action, params):
        if action.startswith("tool_") and action != "tool_final":
            await sync_event.wait()
            return MockObservation("success")
        return MockObservation("success")

    router.execute = sync_execute
    executor = DAGExecutor(tool_router=router)
    plan = {"steps": steps}
    
    exec_task = asyncio.create_task(executor.execute(plan, mock_context))
    
    # Wait a bit to ensure all parallel tasks are blocked waiting
    await asyncio.sleep(0.1)
    
    # Release all tasks simultaneously
    sync_event.set()
    
    result = await asyncio.wait_for(exec_task, timeout=2.0)
    
    assert result["status"] == "success"
    assert "step_final" in result["results"]
    assert len(result["results"]) == 11

