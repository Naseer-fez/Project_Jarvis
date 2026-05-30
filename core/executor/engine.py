"""
core/executor/engine.py
───────────────────────
Asynchronous DAG execution engine with LIFO rollback, retry semantics, and timeouts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Set

from core.context.context import TaskExecutionContext
from core.executor.dag import PlanDAG

logger = logging.getLogger("Jarvis.Executor.Engine")


class DAGExecutor:
    """Executes planned task steps concurrently conforming to dependency constraints."""

    def __init__(self, tool_router: Any, risk_evaluator: Any = None, autonomy_governor: Any = None):
        self.router = tool_router
        self.risk = risk_evaluator
        self.gov = autonomy_governor
        self._rollbacks: List[Callable[[], Any]] = []

    def register_rollback(self, callback: Callable[[], Any]) -> None:
        """Register a LIFO rollback callback."""
        self._rollbacks.append(callback)

    async def execute(self, plan: Dict[str, Any], context: TaskExecutionContext) -> Dict[str, Any]:
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
            sorted_step_ids = dag.topological_sort()
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

        lock = asyncio.Lock()

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

                    observation = await self.router.execute(action, params)

                    if observation.execution_status == "success":
                        context.log(f"Step {step_id} succeeded.")
                        res_dict = observation.to_dict()

                        # Snapshot at end of step (success)
                        context.save_snapshot(step_id=step_id, metadata={"status": "success", "action": action})
                        publish_event("step_completed", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "status": "success", "result": res_dict})

                        # Set up step rollback callback if defined in step schema
                        rollback_def = step.get("rollback")
                        if rollback_def:
                            rb_action = rollback_def.get("action")
                            rb_params = rollback_def.get("params", {})
                            async def rollback_callback():
                                context.log(f"Rolling back step {step_id} via {rb_action}")
                                await self.router.execute(rb_action, rb_params)
                            self.register_rollback(rollback_callback)

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

        async def scheduler_loop():
            while True:
                ready_to_run = []
                async with lock:
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
                        await asyncio.sleep(0.05)
                        continue
                    else:
                        break

                async def run_and_update(sid: str) -> None:
                    try:
                        res = await run_step(sid)
                        async with lock:
                            step_states[sid] = "success"
                            step_results[sid] = res
                            completed_steps.add(sid)
                    except Exception as e:
                        async with lock:
                            step_states[sid] = "failed"
                            step_results[sid] = {"success": False, "error": str(e)}
                            context.log(f"Step {sid} permanently failed: {e}", level="ERROR")

                await asyncio.gather(*(run_and_update(sid) for sid in ready_to_run))

        await scheduler_loop()

        failed_steps = [sid for sid, state in step_states.items() if state == "failed"]
        if failed_steps:
            context.log(
                f"DAG execution halted at failed steps {failed_steps}. Rolling back in LIFO order...", 
                level="ERROR"
            )
            for rb in reversed(self._rollbacks):
                try:
                    await rb()
                except Exception as e:
                    context.log(f"Rollback callback failure: {e}", level="ERROR")
            
            publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "failed", "failed_steps": failed_steps})
            return {"status": "failure", "failed_steps": failed_steps, "results": step_results}

        publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "success"})
        return {"status": "success", "results": step_results}
