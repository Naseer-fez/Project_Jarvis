"""
core/execution/dispatcher.py
Jarvis V3 - Execution Dispatcher
"""

import logging
from typing import Any

from core.agentic.autonomy_policy import AutonomyPolicy, PolicyDecision, PolicyVerdict
from core.tools.system_automation import (
    TOOL_REGISTRY,
    ToolResult,
    async_list_directory,
    async_read_file,
    async_write_file,
    async_delete_file,
    async_launch_application,
    async_execute_shell,
)

logger = logging.getLogger(__name__)

# Risk threshold above which we REQUIRE_CONFIRM regardless of policy mode
CONFIRM_THRESHOLD = 0.5


class DispatchError(Exception):
    pass


class Dispatcher:
    def __init__(
        self,
        autonomy_policy: AutonomyPolicy,
        reflection_engine,          # ReflectionEngine instance
        voice_layer=None,           # Optional; must expose ask_confirm(prompt: str) -> bool
    ):
        self.policy = autonomy_policy
        self.reflection = reflection_engine
        self.voice = voice_layer

        # Map action names to their async callables
        self._tool_map = {
            "list_directory":     self._run_list_directory,
            "read_file":          self._run_read_file,
            "write_file":         self._run_write_file,
            "delete_file":        self._run_delete_file,
            "launch_application": self._run_launch_application,
            "execute_shell":      self._run_execute_shell,
        }

    async def dispatch(self, action: dict[str, Any]) -> ToolResult:
        tool_name: str = action.get("tool", "")
        args: dict = action.get("args", {})
        rationale: str = action.get("rationale", "")

        if tool_name not in self._tool_map:
            result = ToolResult(False, error=f"Unknown tool: '{tool_name}'")
            await self._feed_reflection(tool_name, args, result)
            return result

        risk_score = TOOL_REGISTRY.get(tool_name, 1.0)
        logger.info("Dispatching tool='%s' risk=%.2f rationale='%s'", tool_name, risk_score, rationale)

        # ── Policy gate ───────────────────────
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by AutonomyPolicy or user rejected.")
            await self._feed_reflection(tool_name, args, result)
            return result

        # ── Execute ───────────────────────────
        try:
            result = await self._tool_map[tool_name](args)
        except Exception as exc:
            logger.exception("Unexpected error executing tool '%s'", tool_name)
            result = ToolResult(False, error=f"Dispatcher internal error: {exc}")

        logger.info("Tool '%s' finished success=%s", tool_name, result.success)
        await self._feed_reflection(tool_name, args, result)
        return result

    async def _check_policy(self, tool_name: str, args: dict, risk_score: float) -> bool:
        """
        Returns True if execution is permitted by the AutonomyPolicy.
        """
        # Create a mock context compatible with your AutonomyPolicy rules
        class MinimalContext:
            interrupt_flag = False
            paused = False
            risk_score = risk_score
            confidence_score = 0.99
            def snapshot(self): return {"risk_score": risk_score}

        ctx = MinimalContext()

        decision: PolicyDecision = self.policy.check(context=ctx, action_name=tool_name, params=args)

        if decision.verdict == PolicyVerdict.DENY:
            logger.warning("Policy DENIED tool='%s'. Reason: %s", tool_name, decision.reason)
            return False

        if decision.verdict == PolicyVerdict.ALLOW and risk_score < CONFIRM_THRESHOLD:
            return True

        # If REQUIRE_APPROVAL or risk >= CONFIRM_THRESHOLD
        logger.info("Policy REQUIRE_APPROVAL for tool='%s'", tool_name)
        return await self._ask_voice_confirm(tool_name, args)

    async def _ask_voice_confirm(self, tool_name: str, args: dict) -> bool:
        if self.voice is None:
            logger.warning("No voice layer attached; blocking high-risk tool '%s' by default.", tool_name)
            return False

        summary = _summarise_action(tool_name, args)
        prompt = f"I need your approval. I am about to {summary}. Should I proceed?"

        try:
            confirmed: bool = await self.voice.ask_confirm(prompt)
            if not confirmed:
                logger.info("User rejected action '%s' via voice.", tool_name)
            return confirmed
        except Exception as exc:
            logger.error("Voice confirmation failed: %s — blocking action '%s'.", exc, tool_name)
            return False

    async def _feed_reflection(self, tool_name: str, args: dict, result: ToolResult):
        try:
            payload = {
                "tool": tool_name,
                "args": args,
                **result.to_reflection_payload(),
            }
            if hasattr(self.reflection, "record_action"):
                import asyncio, inspect
                fn = self.reflection.record_action
                if inspect.iscoroutinefunction(fn):
                    await fn(payload)
                else:
                    await asyncio.to_thread(fn, payload)
        except Exception as exc:
            logger.warning("ReflectionEngine update failed: %s", exc)

    async def _run_list_directory(self, args: dict) -> ToolResult:
        return await async_list_directory(args["path"])

    async def _run_read_file(self, args: dict) -> ToolResult:
        return await async_read_file(args["path"])

    async def _run_write_file(self, args: dict) -> ToolResult:
        return await async_write_file(args["path"], args["content"], args.get("overwrite", False))

    async def _run_delete_file(self, args: dict) -> ToolResult:
        return await async_delete_file(args["path"])

    async def _run_launch_application(self, args: dict) -> ToolResult:
        return await async_launch_application(args["target"], args.get("args"))

    async def _run_execute_shell(self, args: dict) -> ToolResult:
        return await async_execute_shell(args["command"], args.get("working_dir"))

def _summarise_action(tool_name: str, args: dict) -> str:
    summaries = {
        "execute_shell":      lambda a: f"run the shell command: {a.get('command', '?')}",
        "write_file":         lambda a: f"write to the file: {a.get('path', '?')}",
        "delete_file":        lambda a: f"permanently delete: {a.get('path', '?')}",
        "launch_application": lambda a: f"launch: {a.get('target', '?')}",
        "read_file":          lambda a: f"read: {a.get('path', '?')}",
        "list_directory":     lambda a: f"list directory: {a.get('path', '?')}",
    }
    return summaries.get(tool_name, lambda a: tool_name)(args)
