"""Execution dispatcher with core-tool and integration-tool routing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.agentic.autonomy_policy import AutonomyPolicy, PolicyDecision, PolicyVerdict
from core.tools.system_automation import (
    TOOL_REGISTRY,
    ToolResult,
    async_delete_file,
    async_execute_shell,
    async_launch_application,
    async_list_directory,
    async_read_file,
    async_write_file,
)

try:
    from integrations.registry import integration_registry
except ImportError:  # optional integration package wiring
    integration_registry = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Risk threshold above which we REQUIRE_CONFIRM regardless of policy mode
CONFIRM_THRESHOLD = 0.5

# Integration fallback risk map (used by policy gate)
INTEGRATION_RISK_REGISTRY: dict[str, float] = {
    "read_emails": 0.1,
    "search_emails": 0.1,
    "read_whatsapp_messages": 0.1,
    "list_calendar_events": 0.1,
    "search_calendar": 0.1,
    "send_email": 0.8,
    "send_whatsapp": 0.8,
    "add_calendar_event": 0.6,
}


class DispatchError(Exception):
    pass


class Dispatcher:
    def __init__(
        self,
        autonomy_policy: AutonomyPolicy,
        reflection_engine,
        voice_layer=None,
    ):
        self.policy = autonomy_policy
        self.reflection = reflection_engine
        self.voice = voice_layer

        self._tool_map = {
            "list_directory": self._run_list_directory,
            "read_file": self._run_read_file,
            "write_file": self._run_write_file,
            "delete_file": self._run_delete_file,
            "launch_application": self._run_launch_application,
            "execute_shell": self._run_execute_shell,
        }

    async def dispatch(self, action: dict[str, Any]) -> ToolResult:
        tool_name: str = action.get("tool", "")
        args: dict = action.get("args", {})
        rationale: str = action.get("rationale", "")

        # Built-in system tools
        if tool_name in self._tool_map:
            risk_score = TOOL_REGISTRY.get(tool_name, 1.0)
            logger.info("Dispatching core tool='%s' risk=%.2f rationale='%s'", tool_name, risk_score, rationale)
            return await self._dispatch_core_tool(tool_name, args, risk_score)

        # Dynamic integration tools
        if integration_registry is not None and tool_name in self._integration_tool_names():
            risk_score = INTEGRATION_RISK_REGISTRY.get(tool_name, 0.6)
            logger.info(
                "Dispatching integration tool='%s' risk=%.2f rationale='%s'",
                tool_name,
                risk_score,
                rationale,
            )
            return await self._dispatch_integration_tool(tool_name, args, risk_score)

        result = ToolResult(False, error=f"Unknown tool: '{tool_name}'")
        await self._feed_reflection(tool_name, args, result)
        return result

    def _integration_tool_names(self) -> set[str]:
        if integration_registry is None:
            return set()
        try:
            return {str(tool.get("name", "")).strip() for tool in integration_registry.get_tools()}
        except Exception:  # noqa: BLE001
            return set()

    async def _dispatch_core_tool(self, tool_name: str, args: dict, risk_score: float) -> ToolResult:
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by AutonomyPolicy or user rejected.")
            await self._feed_reflection(tool_name, args, result)
            return result

        try:
            result = await self._tool_map[tool_name](args)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error executing core tool '%s'", tool_name)
            result = ToolResult(False, error=f"Dispatcher internal error: {exc}")

        logger.info("Core tool '%s' finished success=%s", tool_name, result.success)
        await self._feed_reflection(tool_name, args, result)
        return result

    async def _dispatch_integration_tool(self, tool_name: str, args: dict, risk_score: float) -> ToolResult:
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by AutonomyPolicy or user rejected.")
            await self._feed_reflection(tool_name, args, result)
            return result

        try:
            result_dict = await integration_registry.execute(tool_name, args or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error executing integration tool '%s'", tool_name)
            result_dict = {"success": False, "data": None, "error": str(exc)}

        result = ToolResult(
            success=bool(result_dict.get("success", False)),
            output=str(result_dict.get("data", "")) if result_dict.get("data") is not None else "",
            error=str(result_dict.get("error", "") or ""),
            metadata={"integration": True, "tool": tool_name},
        )

        logger.info("Integration tool '%s' finished success=%s", tool_name, result.success)
        await self._feed_reflection(tool_name, args, result)
        return result

    async def _check_policy(self, tool_name: str, args: dict, risk_score: float) -> bool:
        """Returns True if execution is permitted by the AutonomyPolicy."""

        class MinimalContext:
            def __init__(self, score: float) -> None:
                self.interrupt_flag = False
                self.paused = False
                self.risk_score = score
                self.confidence_score = 0.99

            def snapshot(self) -> dict[str, float]:
                return {"risk_score": self.risk_score}

        ctx = MinimalContext(risk_score)

        decision: PolicyDecision = self.policy.check(context=ctx, action_name=tool_name, params=args)

        if decision.verdict == PolicyVerdict.DENY:
            logger.warning("Policy DENIED tool='%s'. Reason: %s", tool_name, decision.reason)
            return False

        if decision.verdict == PolicyVerdict.ALLOW and risk_score < CONFIRM_THRESHOLD:
            return True

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
        except Exception as exc:  # noqa: BLE001
            logger.error("Voice confirmation failed: %s - blocking action '%s'.", exc, tool_name)
            return False

    async def _feed_reflection(self, tool_name: str, args: dict, result: ToolResult):
        try:
            payload = {
                "tool": tool_name,
                "args": args,
                **result.to_reflection_payload(),
            }
            if hasattr(self.reflection, "record_action"):
                import inspect

                fn = self.reflection.record_action
                if inspect.iscoroutinefunction(fn):
                    await fn(payload)
                else:
                    await asyncio.to_thread(fn, payload)
        except Exception as exc:  # noqa: BLE001
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
        "execute_shell": lambda a: f"run the shell command: {a.get('command', '?')}",
        "write_file": lambda a: f"write to the file: {a.get('path', '?')}",
        "delete_file": lambda a: f"permanently delete: {a.get('path', '?')}",
        "launch_application": lambda a: f"launch: {a.get('target', '?')}",
        "read_file": lambda a: f"read: {a.get('path', '?')}",
        "list_directory": lambda a: f"list directory: {a.get('path', '?')}",
        "send_email": lambda a: f"send an email to {a.get('to', '?')}",
        "send_whatsapp": lambda a: f"send a WhatsApp message to {a.get('to', '?')}",
        "add_calendar_event": lambda a: f"add calendar event '{a.get('title', '?')}'",
    }
    return summaries.get(tool_name, lambda a: tool_name)(args)
