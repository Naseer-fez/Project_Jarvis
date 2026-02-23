# core/execution/dispatcher.py
# Safe task dispatcher stub — routes tasks to the execution layer.

from typing import Any, Optional


def dispatch(task: Any, mode: str = "sync") -> dict:
    """
    Dispatch a task to the execution layer.

    Args:
        task:  The task payload (dict, string, or object).
        mode:  Execution mode hint — 'sync' or 'async' (not yet implemented).

    Returns:
        A result dictionary with status information.
    """
    try:
        if task is None:
            return {
                "status": "skipped",
                "reason": "null task provided",
                "task": None,
            }

        task_id = getattr(task, "id", None) or (
            task.get("id") if isinstance(task, dict) else None
        )

        # TODO: replace with real routing / queue logic
        return {
            "status": "accepted",
            "task_id": task_id,
            "mode": mode,
            "reason": "dispatcher stub — task queued for future execution",
        }

    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "task": None,
        }


async def dispatch_async(task: Any) -> dict:
    """
    Async variant of dispatch. Currently delegates to the sync stub.

    Args:
        task: The task payload.

    Returns:
        A result dictionary with status information.
    """
    return dispatch(task, mode="async")
