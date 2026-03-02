"""Execution dispatcher with core-tool and dynamic-integration routing."""

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
except Exception:  # noqa: BLE001
    integration_registry = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CONFIRM_THRESHOLD = 0.5

INTEGRATION_RISK_REGISTRY: dict[str, float] = {
    "read_emails": 0.1,
    "search_emails": 0.1,
    "list_events": 0.1,
    "get_current_weather": 0.1,
    "send_email": 0.8,
    "send_whatsapp": 0.8,
    "add_event": 0.6,
}


class DispatchError(Exception):
    pass


class Dispatcher:
    def __init__(self, autonomy_policy: AutonomyPolicy, reflection_engine, voice_layer=None):
        self.policy = autonomy_policy
        self.reflection = reflection_engine
        self.voice = voice_layer

        self._core_tools = {
            "list_directory": self._run_list_directory,
            "read_file": self._run_read_file,
            "write_file": self._run_write_file,
            "delete_file": self._run_delete_file,
            "launch_application": self._run_launch_application,
            "execute_shell": self._run_execute_shell,
        }

    async def dispatch(self, action: dict[str, Any]) -> ToolResult:
        tool_name = str(action.get("tool", "")).strip()
        args = action.get("args", {}) or {}
        rationale = str(action.get("rationale", "")).strip()

        # 1) Built-in core tool registry first.
        if tool_name in TOOL_REGISTRY and tool_name in self._core_tools:
            risk_score = float(TOOL_REGISTRY.get(tool_name, 1.0))
            logger.info("Dispatch core tool='%s' risk=%.2f rationale='%s'", tool_name, risk_score, rationale)
            return await self._dispatch_core_tool(tool_name, args, risk_score)

        # 2) Dynamic integrations second.
        if self._has_integration_tool(tool_name):
            risk_score = float(INTEGRATION_RISK_REGISTRY.get(tool_name, 0.6))
            logger.info(
                "Dispatch integration tool='%s' risk=%.2f rationale='%s'",
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
            return {
                str(tool.get("name", "")).strip()
                for tool in integration_registry.get_tools()
                if isinstance(tool, dict)
            }
        except Exception:  # noqa: BLE001
            return set()

    def _has_integration_tool(self, tool_name: str) -> bool:
        if integration_registry is None:
            return False

        getter = getattr(integration_registry, "get_tool", None)
        if callable(getter):
            try:
                return getter(tool_name) is not None
            except Exception:  # noqa: BLE001
                pass

        return tool_name in self._integration_tool_names()

    async def _dispatch_core_tool(self, tool_name: str, args: dict[str, Any], risk_score: float) -> ToolResult:
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by policy or user rejection")
            await self._feed_reflection(tool_name, args, result)
            return result

        try:
            result = await self._core_tools[tool_name](args)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Core tool '%s' failed", tool_name)
            result = ToolResult(False, error=f"Dispatcher error: {exc}")

        await self._feed_reflection(tool_name, args, result)
        return result

    async def _dispatch_integration_tool(self, tool_name: str, args: dict[str, Any], risk_score: float) -> ToolResult:
        allowed = await self._check_policy(tool_name, args, risk_score)
        if not allowed:
            result = ToolResult(False, error=f"Action '{tool_name}' blocked by policy or user rejection")
            await self._feed_reflection(tool_name, args, result)
            return result

        try:
            payload = await integration_registry.execute(tool_name, args)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            payload = {"success": False, "data": None, "error": str(exc)}

        result = ToolResult(
            success=bool(payload.get("success", False)),
            output=str(payload.get("data", "")) if payload.get("data") is not None else "",
            error=str(payload.get("error", "") or ""),
            metadata={"integration": True, "tool": tool_name},
        )

        await self._feed_reflection(tool_name, args, result)
        return result

    async def _check_policy(self, tool_name: str, args: dict[str, Any], risk_score: float) -> bool:
        class Context:
            def __init__(self, score: float) -> None:
                self.interrupt_flag = False
                self.paused = False
                self.risk_score = score
                self.confidence_score = 0.99

            def snapshot(self) -> dict[str, float]:
                return {"risk_score": self.risk_score}

        context = Context(risk_score)
        decision: PolicyDecision = self.policy.check(context=context, action_name=tool_name, params=args)

        if decision.verdict == PolicyVerdict.DENY:
            logger.warning("Policy denied tool '%s': %s", tool_name, decision.reason)
            return False

        if decision.verdict == PolicyVerdict.ALLOW and risk_score < CONFIRM_THRESHOLD:
            return True

        return await self._ask_voice_confirm(tool_name, args)

    async def _ask_voice_confirm(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self.voice is None:
            logger.warning("No voice layer attached; blocking high-risk action '%s'", tool_name)
            return False

        summary = _summarise_action(tool_name, args)
        prompt = f"I need approval before I {summary}. Proceed?"
        try:
            return bool(await self.voice.ask_confirm(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Voice confirmation failed for '%s': %s", tool_name, exc)
            return False

    async def _feed_reflection(self, tool_name: str, args: dict[str, Any], result: ToolResult) -> None:
        if self.reflection is None:
            return
        payload = {
            "tool": tool_name,
            "args": args,
            **result.to_reflection_payload(),
        }
        try:
            fn = getattr(self.reflection, "record_action", None)
            if fn is None:
                return
            if asyncio.iscoroutinefunction(fn):
                await fn(payload)
            else:
                await asyncio.to_thread(fn, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Reflection hook failed: %s", exc)

    async def _run_list_directory(self, args: dict[str, Any]) -> ToolResult:
        return await async_list_directory(args["path"])

    async def _run_read_file(self, args: dict[str, Any]) -> ToolResult:
        return await async_read_file(args["path"])

    async def _run_write_file(self, args: dict[str, Any]) -> ToolResult:
        return await async_write_file(args["path"], args["content"], bool(args.get("overwrite", False)))

    async def _run_delete_file(self, args: dict[str, Any]) -> ToolResult:
        return await async_delete_file(args["path"])

    async def _run_launch_application(self, args: dict[str, Any]) -> ToolResult:
        return await async_launch_application(args["target"], args.get("args"))

    async def _run_execute_shell(self, args: dict[str, Any]) -> ToolResult:
        return await async_execute_shell(args["command"], args.get("working_dir"))


def _summarise_action(tool_name: str, args: dict[str, Any]) -> str:
    summaries = {
        "execute_shell": lambda x: f"run shell command '{x.get('command', '')}'",
        "write_file": lambda x: f"write file '{x.get('path', '')}'",
        "delete_file": lambda x: f"delete file '{x.get('path', '')}'",
        "launch_application": lambda x: f"launch '{x.get('target', '')}'",
        "send_email": lambda x: f"send an email to {x.get('to', '')}",
        "send_whatsapp": lambda x: f"send a WhatsApp message to {x.get('to', '')}",
        "add_event": lambda x: f"add calendar event '{x.get('title', '')}'",
    }
    return summaries.get(tool_name, lambda _: f"run '{tool_name}'")(args)


__all__ = ["DispatchError", "Dispatcher"]
