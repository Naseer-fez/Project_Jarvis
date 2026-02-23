"""
TOOL_ROUTER_PATCH.py  ← NOT a real file; paste these snippets into tool_router.py
===========================================================================

These are the ONLY additions needed to tool_router.py to support integrations/.
Nothing in core/ needs to change.

IMPORTANT: Do not modify the state machine, the main event loop, or any
           existing dispatch logic.  Only add the marked blocks below.
===========================================================================
"""

# ---------------------------------------------------------------------------
# PATCH 1 — Add near the top of tool_router.py, with the other imports
# ---------------------------------------------------------------------------
#
# from integrations import get_tool, list_schemas   # ← ADD THIS LINE
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PATCH 2 — Inside the existing route_tool() / dispatch() function,
#            add this branch alongside your existing tool handlers.
#
#  The AutonomyGovernor and RiskEvaluator checks come FIRST, as always.
# ---------------------------------------------------------------------------

async def _example_route_tool_addition(
    tool_name: str,
    tool_args: dict,
    autonomy_governor,      # your existing AutonomyGovernor instance
) -> str:
    """
    Drop-in addition for the existing route_tool() function.
    Shows exactly where to insert integration dispatch.
    """

    # ---- EXISTING: look up built-in tools first (do not change this) -----
    # if tool_name in BUILTIN_TOOLS:
    #     ...

    # ---- NEW: fall through to integration registry -----------------------
    integration = get_tool(tool_name)

    if integration is None:
        return f"{{'error': 'Unknown tool: {tool_name}'}}"

    # ---- Autonomy / risk gate (NEVER bypass this) ------------------------
    can_run = autonomy_governor.can_execute(
        tool_name=tool_name,
        risk_level=integration.risk_level.value,
    )
    if not can_run:
        return (
            f"{{'error': 'Tool {tool_name!r} blocked by AutonomyGovernor. "
            f"Required level: {integration.risk_level.value}'}}"
        )

    # ---- Execute with timeout guard --------------------------------------
    import asyncio
    try:
        result = await asyncio.wait_for(
            integration.execute(**tool_args),
            timeout=15.0,                         # seconds; tune as needed
        )
    except asyncio.TimeoutError:
        return f"{{'error': 'Tool {tool_name!r} execution timed out.'}}"

    return result.to_llm_string()


# ---------------------------------------------------------------------------
# PATCH 3 — Optionally expose schemas to task_planner at startup
# ---------------------------------------------------------------------------
#
# In your task_planner.py or wherever you build the LLM's tool context:
#
#   from integrations import list_schemas
#   external_tool_schemas = list_schemas()
#   # merge into your existing tool schema list before building the prompt
#
# ---------------------------------------------------------------------------

