import uuid
import logging
from typing import TYPE_CHECKING
from core.controller.request_rules import is_active_window_request, is_explicit_web_search
from core.desktop.shortcuts import handle_desktop_command, plan_desktop_command
from core.controller.web_search import handle_web_search

if TYPE_CHECKING:
    from core.controller_v2 import JarvisControllerV2

logger = logging.getLogger(__name__)


def register_intent_routes(ctx: "JarvisControllerV2") -> None:
    async def handle_status(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return f"Session: {ctx.session_id} | Memory Mode: {ctx.memory.mode}"
    ctx.intent_router.register(lambda _l, _u, c: _l == "status", handle_status)

    async def handle_help(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return "Commands: status, help, exit, remember <fact>, what's <query>, open <app>, search <query> in <browser>"
    ctx.intent_router.register(lambda _l, _u, c: _l == "help", handle_help)

    async def handle_automation(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        am = getattr(ctx, "automation_manager", None)
        la = getattr(am, "live_automation", None) if am else None
        if la is None:
            return None
        if lowered == "automation status":
            status_info = la.status()
            return f"{la.status_line()}\nDrop Root: {status_info.get('drop_root')}\nCommands Dir: {status_info.get('commands_dir')}\nRAG Dir: {status_info.get('rag_dir')}"
        elif lowered == "automation scan":
            scan_res = await la.force_scan()
            return f"Scan completed: commands={scan_res.get('commands_processed', 0)} files={scan_res.get('files_ingested', 0)} chunks={scan_res.get('chunks_ingested', 0)}"
        elif lowered.startswith("rag search "):
            query = user_input[len("rag search "):].strip()
            res = await la.search_rag(query)
            return str(res) if res is not None else None
        return None
    ctx.intent_router.register(lambda _l, _u, c: (getattr(getattr(c, "automation_manager", None), "live_automation", None) is not None) and (_l in ("automation status", "automation scan") or _l.startswith("rag search ")), handle_automation)

    async def handle_goal(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return await ctx._handle_goal_intent(lowered, user_input)
    # Always run, returns None if not matched inside
    ctx.intent_router.register(lambda _l, _u, c: True, handle_goal)

    async def handle_pref(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return await ctx._handle_preference_intent(lowered, user_input)
    ctx.intent_router.register(lambda _l, _u, c: True, handle_pref)

    async def handle_active_window(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        if is_active_window_request(lowered):
            obs = await ctx.desktop_observer.observe()
            title = obs.active_window.get("title", "")
            return f"The active window is: {title}"
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_active_window)

    async def handle_desktop_plan(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        desktop_plan = plan_desktop_command(user_input)
        if desktop_plan is not None:
            if not ctx._app_launch_enabled:
                return ctx._app_launch_disabled_message()
            return await handle_desktop_command(user_input)
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_desktop_plan)

    async def handle_desktop_disabled(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        if ctx._looks_like_desktop_control_request(lowered) and not ctx._gui_automation_enabled:
            return ctx._desktop_control_disabled_message()
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_desktop_disabled)

    async def handle_explicit_web(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        if is_explicit_web_search(lowered):
            ctx._dashboard_update(state="EXECUTE", last_input=user_input)

            web_response = await handle_web_search(
                user_input=user_input, 
                trace_id=uuid.uuid4().hex[:8], 
                memory=ctx.memory, 
                llm=ctx.llm, 
                model_router=ctx.model_router, 
                profile=ctx.profile
            )
            if web_response:
                await ctx.memory.store_conversation(user_input, web_response, ctx.session_id)
                return web_response
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_explicit_web)

    async def handle_agentic(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        classification = getattr(ctx, "current_classification", {})
        if not classification.get("skip_planner", False) and classification.get("route") in ("planner", "premium", "full stack"):
            ctx._dashboard_update(state="PLANNING", last_input=user_input)
            
            # Let task_planner also know about the complexity
            plan = await ctx.task_planner.plan(user_input)
            if plan and plan.get("tools_required"):
                ctx._dashboard_update(state="EXECUTE", last_input=user_input)
                
                task_sm = ctx.container.resolve("state_machine") if ctx.container else None
                if not task_sm:
                    raise RuntimeError("state_machine not found in container")
                
                def _update_dash_state(_old, new):
                    ctx._dashboard_update(state=new.value)
                task_sm.add_listener(_update_dash_state)
                
                try:
                    context = ctx.container.resolve(
                        "task_execution_context",
                        task_id=uuid.uuid4().hex[:8],
                        trace_id=uuid.uuid4().hex[:8],
                        state_machine=task_sm,
                    )
                    
                    # Only build massive context if it's high complexity
                    if classification.get("complexity", 0.5) > 0.5:
                        context_block = await ctx.memory.build_context_block(user_input)
                        context.set("context_block", context_block)
                    else:
                        context.set("context_block", "")
                    
                    trace = await ctx.agent_loop.run(
                        goal=user_input,
                        context=context,
                    )
                    return str(trace.final_response)
                finally:
                    if hasattr(task_sm, "remove_listener"):
                        task_sm.remove_listener(_update_dash_state)
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_agentic)
