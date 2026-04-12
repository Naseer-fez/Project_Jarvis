"""Simple, safe desktop shortcuts for common open/search commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from core.tools.system_automation import async_launch_application

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DesktopCommandPlan:
    app_label: str
    primary_target: str
    primary_args: list[str] | None = None
    response_text: str = ""


def _supported_apps_message() -> str:
    return (
        "I can open: Microsoft Edge, Notepad, Calculator, or the Jarvis project folder."
    )


def _extract_search_query(lowered: str, original: str) -> str:
    for marker in ("search ", "find "):
        idx = lowered.find(marker)
        if idx == -1:
            continue
        query = original[idx + len(marker):].strip()
        for suffix in (" in that browser", " on that browser", " in the browser"):
            if query.lower().endswith(suffix):
                query = query[: -len(suffix)].strip()
        return query.strip(" .")
    return ""


def plan_desktop_command(user_input: str) -> DesktopCommandPlan | None:
    text = str(user_input or "").strip()
    lowered = text.lower()

    if not text:
        return None

    if "project folder" in lowered and "jarvis" in lowered:
        return DesktopCommandPlan(
            app_label="Jarvis project folder",
            primary_target="explorer.exe",
            primary_args=[str(PROJECT_ROOT)],
            response_text="Opened the Jarvis project folder.",
        )

    if "any app" in lowered and ("open" in lowered or "access" in lowered):
        return DesktopCommandPlan(
            app_label="Notepad",
            primary_target="notepad.exe",
            primary_args=None,
            response_text="Opened Notepad.",
        )

    if "edge" in lowered and any(token in lowered for token in ("search ", "find ")):
        query = _extract_search_query(lowered, text)
        if query:
            return DesktopCommandPlan(
                app_label="Microsoft Edge",
                primary_target="msedge.exe",
                primary_args=[f"https://www.bing.com/search?q={quote_plus(query)}"],
                response_text=f'Opened Microsoft Edge and searched for "{query}".',
            )

    if any(alias in lowered for alias in ("edge", "microst edge", "microsoft edge")) and (
        lowered.startswith("open ")
        or lowered.startswith("go to ")
        or lowered.startswith("access ")
        or lowered.startswith("acces ")
    ):
        return DesktopCommandPlan(
            app_label="Microsoft Edge",
            primary_target="msedge.exe",
            primary_args=None,
            response_text="Opened Microsoft Edge.",
        )

    if lowered.startswith("open notepad") or lowered.startswith("open note pad"):
        return DesktopCommandPlan(
            app_label="Notepad",
            primary_target="notepad.exe",
            primary_args=None,
            response_text="Opened Notepad.",
        )

    if lowered.startswith("open calculator") or lowered.startswith("open calc"):
        return DesktopCommandPlan(
            app_label="Calculator",
            primary_target="calc.exe",
            primary_args=None,
            response_text="Opened Calculator.",
        )

    if lowered.startswith(("open ", "go to ", "access ", "acces ")):
        return DesktopCommandPlan(
            app_label="Unsupported app",
            primary_target="",
            primary_args=None,
            response_text=_supported_apps_message(),
        )

    return None


async def handle_desktop_command(user_input: str) -> str | None:
    plan = plan_desktop_command(user_input)
    if plan is None:
        return None
    if not plan.primary_target:
        return plan.response_text

    result = await async_launch_application(plan.primary_target, plan.primary_args)
    if getattr(result, "success", False):
        return plan.response_text
    error = getattr(result, "error", "") or "Unknown launch failure."
    return f"I couldn't open {plan.app_label}: {error}"


__all__ = ["DesktopCommandPlan", "handle_desktop_command", "plan_desktop_command"]
