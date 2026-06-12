"""Simple, safe desktop shortcuts for common open/search commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from core.desktop.actions import DesktopActionExecutor
from core.desktop.contracts import (
    DesktopAction,
    DesktopActionType,
)
from core.desktop.mission import (
    DesktopMissionExecutor,
    DesktopMissionStatus,
)
from core.desktop.observation import DesktopObserver
from core.tools.system_automation import async_launch_application

INTERACTIVE_VERBS_PATTERN = re.compile(
    r"\b("
    # English
    r"write|type|fill|enter|input|select|click|double-?click|right-?click|middle-?click|press|hotkey|key|drag|drop|scroll|typewrite|keystroke|tap|check|tick|focus|copy|paste|cut|clipboard"
    # Spanish
    r"|escribir|teclear|rellenar|introducir|pulsar|presionar|clic|pinchar|seleccionar|arrastrar|pegar|copiar"
    # French
    r"|écrire|ecrire|taper|saisir|remplir|cliquer|appuyer|presser|sélectionner|selectionner|glisser|coller|copier"
    # German
    r"|schreiben|tippen|eingeben|ausfüllen|ausfullen|klicken|drücken|drucken|auswählen|auswahlen|ziehen|einfügen|einfugen|kopieren"
    r")\b",
    re.IGNORECASE
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DesktopCommandPlan:
    app_label: str
    primary_target: str
    primary_args: list[str] | None = None
    response_text: str = ""


def _supported_apps_message() -> str:
    return (
        "I can open: Microsoft Edge, Visual Studio Code, Notepad, Calculator, or the Jarvis project folder."
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

    # If the user request involves keyboard/mouse interactions, bypass the simple launcher
    # and let the full agentic loop plan the execution.
    if INTERACTIVE_VERBS_PATTERN.search(lowered):
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
        "open" in lowered
        or "go to" in lowered
        or "access" in lowered
        or "acces" in lowered
        or "launch" in lowered
    ):
        return DesktopCommandPlan(
            app_label="Microsoft Edge",
            primary_target="msedge.exe",
            primary_args=None,
            response_text="Opened Microsoft Edge.",
        )

    if ("notepad" in lowered or "note pad" in lowered) and ("open" in lowered or "launch" in lowered):
        return DesktopCommandPlan(
            app_label="Notepad",
            primary_target="notepad.exe",
            primary_args=None,
            response_text="Opened Notepad.",
        )

    if any(alias in lowered for alias in ("vscode", "vs code", "visual studio code")) and ("open" in lowered or "launch" in lowered):
        return DesktopCommandPlan(
            app_label="Visual Studio Code",
            primary_target="code",
            primary_args=[str(PROJECT_ROOT)],
            response_text="Opened Visual Studio Code.",
        )

    if ("calculator" in lowered or "calc" in lowered) and ("open" in lowered or "launch" in lowered):
        return DesktopCommandPlan(
            app_label="Calculator",
            primary_target="calc.exe",
            primary_args=None,
            response_text="Opened Calculator.",
        )

    if "open " in lowered or "go to " in lowered or "access " in lowered or "acces " in lowered or "launch " in lowered:
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

    async def _launch_handler(target: str, args: list[str] | None = None):
        return await async_launch_application(target, args)

    action = DesktopAction(
        action_type=DesktopActionType.LAUNCH_APP,
        description=f"Open {plan.app_label}",
        params={"target": plan.primary_target, "args": plan.primary_args},
        requires_approval=False,
        metadata={"source": "desktop_shortcut", "app_label": plan.app_label},
    )
    action_executor = DesktopActionExecutor(
        action_handlers={DesktopActionType.LAUNCH_APP: _launch_handler}
    )
    mission_executor = DesktopMissionExecutor(
        action_executor=action_executor,
        observer=DesktopObserver(),
        max_retries=0,
        min_confidence=0.0,
    )
    record = await mission_executor.run(
        goal=user_input,
        actions=[action],
        plan_summary=plan.response_text,
    )
    if record.status == DesktopMissionStatus.SUCCEEDED:
        return plan.response_text

    step = record.steps[-1] if record.steps else None
    result = step.result if step else {}
    error = (
        str(result.get("error", "") or "")
        if isinstance(result, dict)
        else ""
    )
    if not error and step is not None:
        error = step.error
    error = error or "Unknown launch failure."
    return f"I couldn't open {plan.app_label}: {error}"
