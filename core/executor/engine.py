"""
core/executor/engine.py
───────────────────────
Asynchronous DAG execution engine with LIFO rollback, retry semantics, and timeouts.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Set

from core.context.context import TaskExecutionContext
from core.executor.dag import PlanDAG

logger = logging.getLogger("Jarvis.Executor.Engine")


class DAGExecutor:
    """Executes planned task steps concurrently conforming to dependency constraints."""

    def __init__(self, tool_router: Any, risk_evaluator: Any = None, autonomy_governor: Any = None):
        self.router = tool_router
        self.risk = risk_evaluator
        self.gov = autonomy_governor

    async def execute(self, plan: Dict[str, Any], context: TaskExecutionContext) -> Dict[str, Any]:
        rollbacks: Dict[str, Callable[[], Any]] = {}

        def register_rollback(step_id: str, step_def: Dict[str, Any]):
            rollback_def = step_def.get("rollback")
            if rollback_def:
                rb_action = rollback_def.get("action")
                rb_params = rollback_def.get("params", {})
                async def rollback_callback():
                    context.log(f"Rolling back step {step_id} via {rb_action}")
                    sig_rb = inspect.signature(self.router.execute)
                    if "context" in sig_rb.parameters:
                        await self.router.execute(rb_action, rb_params, context=context)
                    else:
                        await self.router.execute(rb_action, rb_params)
                rollbacks[step_id] = rollback_callback

        steps = plan.get("steps", [])
        if not steps:
            return {"status": "success", "message": "No steps to execute."}

        # Helper to publish events
        def publish_event(event_type: str, data: Any):
            eb = getattr(context, "event_bus", None)
            if eb:
                eb.publish(event_type, data)

        publish_event("task_started", {"trace_id": context.trace_id, "task_id": context.task_id, "plan": plan})

        dag = PlanDAG(steps)
        try:
            topo_order = dag.topological_sort()
        except Exception as e:
            context.log(f"Topological sort failed: {e}", level="ERROR")
            publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "failed", "error": str(e)})
            return {"status": "failure", "error": str(e)}

        # Track execution states of steps
        step_states = {step_id: "pending" for step_id in dag.step_map}
        
        # Load or initialize step_results in context.variables for replay capability
        if "_step_results" not in context.variables:
            context.variables["_step_results"] = {}
        step_results = context.variables["_step_results"]
        
        completed_steps: Set[str] = set()
        
        # In replay mode, check if step is already completed
        replay_active = context.get("_replay_active", False)
        if replay_active:
            for sid, res in step_results.items():
                if isinstance(res, dict) and res.get("success", True) is not False:
                    step_states[sid] = "success"
                    completed_steps.add(sid)
                    context.log(f"Replay: loaded step {sid} status as completed from snapshot.")
                    if sid in dag.step_map:
                        register_rollback(sid, dag.step_map[sid])

        cond = asyncio.Condition()
        # Build incoming dependency tracking dictionary
        dependencies: Dict[str, Set[str]] = {step_id: set() for step_id in dag.step_map}
        for step_id, step in dag.step_map.items():
            depends_on = step.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dep in depends_on:
                dep_str = str(dep)
                if dep_str in dag.step_map:
                    dependencies[step_id].add(dep_str)

        async def run_step(step_id: str) -> Dict[str, Any]:
            step = dag.step_map[step_id]
            action = step.get("action") or step.get("tool")
            params = step.get("params", {})
            retry_count = int(step.get("retry_count", 0))

            # Replay mocking check
            if replay_active and step_id in step_results:
                prior_res = step_results[step_id]
                if isinstance(prior_res, dict) and prior_res.get("success", True) is not False:
                    context.log(f"Replay: mocking execution of step {step_id} with prior successful result.")
                    return prior_res

            if self.gov and action:
                allowed, reason = self.gov.can_execute(action)
                if not allowed:
                    raise RuntimeError(f"Autonomy Governor blocked '{action}': {reason}")

            backoff = 1.0
            for attempt in range(retry_count + 1):
                try:
                    context.log(
                        f"Step {step_id}: executing '{action}' (attempt {attempt + 1})"
                    )
                    # Snapshot at start of step
                    context.save_snapshot(step_id=step_id, metadata={"status": "running", "action": action, "attempt": attempt + 1})
                    publish_event("step_executing", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "action": action})

                    sig = inspect.signature(self.router.execute)
                    if "context" in sig.parameters:
                        observation = await self.router.execute(action, params, context=context)
                    else:
                        observation = await self.router.execute(action, params)

                    if observation.execution_status == "success":
                        context.log(f"Step {step_id} succeeded.")
                        res_dict = observation.to_dict()

                        # Snapshot at end of step (success)
                        context.save_snapshot(step_id=step_id, metadata={"status": "success", "action": action})
                        publish_event("step_completed", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "status": "success", "result": res_dict})

                        # Set up step rollback callback if defined in step schema
                        register_rollback(step_id, step)

                        return res_dict
                    else:
                        raise RuntimeError(observation.error_message or "Tool execution failed")
                except Exception as exc:
                    if attempt < retry_count:
                        context.log(
                            f"Step {step_id} failed: {exc}. Retrying in {backoff}s...", 
                            level="WARNING"
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                    else:
                        # Snapshot at end of step (failure)
                        context.save_snapshot(step_id=step_id, metadata={"status": "failed", "action": action, "error": str(exc)})
                        publish_event("step_completed", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "status": "failed", "error": str(exc)})
                        raise exc

        running_tasks: set[asyncio.Task] = set()

        async def run_step_task(sid: str) -> None:
            try:
                res = await run_step(sid)
                async with cond:
                    step_states[sid] = "success"
                    step_results[sid] = res
                    completed_steps.add(sid)
            except asyncio.CancelledError:
                async with cond:
                    step_states[sid] = "cancelled"
                    step_results[sid] = {"success": False, "error": "Cancelled"}
                    context.log(f"Step {sid} cancelled.", level="INFO")
                raise
            except Exception as e:
                async with cond:
                    step_states[sid] = "failed"
                    step_results[sid] = {"success": False, "error": str(e)}
                    context.log(f"Step {sid} permanently failed: {e}", level="ERROR")
            except BaseException as e:
                async with cond:
                    step_states[sid] = "failed"
                    step_results[sid] = {"success": False, "error": f"BaseException: {type(e).__name__}"}
                    context.log(f"Step {sid} aborted due to critical error: {e}", level="ERROR")
                raise
            finally:
                async with cond:
                    cond.notify_all()

        async def scheduler_loop():
            while True:
                ready_to_run = []
                async with cond:
                    if len(completed_steps) == len(dag.step_map):
                        break

                    # Halt everything if any step failed
                    if any(state == "failed" for state in step_states.values()):
                        break

                    for step_id in dag.step_map:
                        if step_states[step_id] == "pending":
                            deps = dependencies[step_id]
                            if deps.issubset(completed_steps):
                                step_states[step_id] = "running"
                                ready_to_run.append(step_id)

                    if not ready_to_run:
                        if any(state == "running" for state in step_states.values()):
                            await cond.wait()
                            continue
                        else:
                            break

                for sid in ready_to_run:
                    task = asyncio.create_task(run_step_task(sid))
                    running_tasks.add(task)
                    task.add_done_callback(running_tasks.discard)

        try:
            await scheduler_loop()
        finally:
            if running_tasks:
                # Cancel remaining tasks before waiting to prevent infinite hangs on failure
                tasks_to_await = list(running_tasks)
                for t in tasks_to_await:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks_to_await, return_exceptions=True)

        failed_steps = [sid for sid, state in step_states.items() if state == "failed"]
        if failed_steps:
            context.log(
                f"DAG execution halted at failed steps {failed_steps}. Rolling back in reverse topological order...", 
                level="ERROR"
            )
            for sid in reversed(topo_order):
                if sid in rollbacks:
                    try:
                        await rollbacks[sid]()
                    except Exception as e:
                        context.log(f"Rollback callback failure for step {sid}: {e}", level="ERROR")
            
            publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "failed", "failed_steps": failed_steps})
            return {"status": "failure", "failed_steps": failed_steps, "results": step_results}

        publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "success"})
        return {"status": "success", "results": step_results}
