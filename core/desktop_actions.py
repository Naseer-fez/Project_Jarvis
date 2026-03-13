"""Helpers for explicit desktop action requests on the CLI path."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from core.tools.system_automation import async_launch_application


@dataclass(frozen=True)
class DesktopApp:
    label: str
    target: str
    aliases: tuple[str, ...]
    is_browser: bool = False
    uri_scheme: str | None = None


@dataclass(frozen=True)
class DesktopCommandPlan:
    app_label: str
    primary_target: str
    primary_args: tuple[str, ...] = ()
    success_message: str = ""
    fallback_target: str | None = None
    fallback_args: tuple[str, ...] = ()


_APPS: tuple[DesktopApp, ...] = (
    DesktopApp(
        label="Microsoft Edge",
        target="msedge.exe",
        aliases=("microsoft edge", "microst edge", "ms edge", "edge"),
        is_browser=True,
        uri_scheme="microsoft-edge:",
    ),
    DesktopApp(
        label="Google Chrome",
        target="chrome.exe",
        aliases=("google chrome", "chrome"),
        is_browser=True,
    ),
    DesktopApp(
        label="Firefox",
        target="firefox.exe",
        aliases=("mozilla firefox", "firefox"),
        is_browser=True,
    ),
    DesktopApp(
        label="Notepad",
        target="notepad.exe",
        aliases=("notepad",),
    ),
    DesktopApp(
        label="Calculator",
        target="calc.exe",
        aliases=("calculator", "calc"),
    ),
    DesktopApp(
        label="File Explorer",
        target="explorer.exe",
        aliases=("file explorer", "explorer"),
    ),
    DesktopApp(
        label="Settings",
        target="ms-settings:",
        aliases=("settings", "windows settings"),
    ),
)

_ALIAS_TO_APP: dict[str, DesktopApp] = {
    alias: app
    for app in _APPS
    for alias in app.aliases
}

_SEARCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:open|launch|start|go to|access|acces|use)\s+(?P<browser>.+?)\s+(?:and\s+)?(?:search(?:\s+for)?|find|look\s+up)\s+(?P<query>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:search(?:\s+for)?|find|look\s+up)\s+(?P<query>.+?)\s+(?:in|on|using)\s+(?P<browser>.+)$",
        re.IGNORECASE,
    ),
)

_OPEN_PATTERN = re.compile(r"^(?:open|launch|start|access|acces|use)\s+(?P<app>.+)$", re.IGNORECASE)
_DEFAULT_APP = _ALIAS_TO_APP["notepad"]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_FOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:go to|open|show|take me to)\s+(?P<location>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<location>.+?)\s+(?:folder|directory)$", re.IGNORECASE),
)


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_app_phrase(text: str) -> str:
    cleaned = _normalize_text(text)
    cleaned = re.sub(r"^(?:the|app|application|browser)\s+", "", cleaned)
    cleaned = re.sub(r"\s+(?:app|application|browser)$", "", cleaned)
    return cleaned.strip()


def _supported_apps_text(*, browsers_only: bool = False) -> str:
    labels: list[str] = []
    for app in _APPS:
        if browsers_only and not app.is_browser:
            continue
        if app.label not in labels:
            labels.append(app.label)
    return ", ".join(labels)


def resolve_desktop_app(text: str, *, browsers_only: bool = False) -> DesktopApp | None:
    cleaned = _clean_app_phrase(text)
    if not cleaned:
        return None

    exact = _ALIAS_TO_APP.get(cleaned)
    if exact and (not browsers_only or exact.is_browser):
        return exact

    alias_pool = [
        alias
        for alias, app in _ALIAS_TO_APP.items()
        if not browsers_only or app.is_browser
    ]
    closest = difflib.get_close_matches(cleaned, alias_pool, n=1, cutoff=0.72)
    if not closest:
        return None
    return _ALIAS_TO_APP[closest[0]]


def _build_search_plan(app: DesktopApp, query: str) -> DesktopCommandPlan:
    search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
    fallback_target = f"{app.uri_scheme}{search_url}" if app.uri_scheme else search_url
    return DesktopCommandPlan(
        app_label=app.label,
        primary_target=app.target,
        primary_args=(search_url,),
        success_message=f'Opened {app.label} and searched for "{query}".',
        fallback_target=fallback_target,
    )


def _build_launch_plan(app: DesktopApp) -> DesktopCommandPlan:
    return DesktopCommandPlan(
        app_label=app.label,
        primary_target=app.target,
        success_message=f"Opened {app.label}.",
        fallback_target=app.uri_scheme,
    )


def _strip_search_suffix(query: str) -> str:
    stripped = query or ""
    stripped = re.sub(r"\s+(?:in|on)\s+that\s+browser\s*$", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+(?:in|on)\s+(?:the\s+)?browser\s*$", "", stripped, flags=re.IGNORECASE)
    return stripped.strip(" .?!")


def _build_project_folder_plan() -> DesktopCommandPlan:
    return DesktopCommandPlan(
        app_label="Jarvis project folder",
        primary_target="explorer.exe",
        primary_args=(str(_PROJECT_ROOT),),
        success_message="Opened the Jarvis project folder.",
    )


def _is_project_folder_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(
        phrase in normalized
        for phrase in (
            "jarvis project folder",
            "jarvis folder",
            "project folder",
            "project directory",
        )
    )


def plan_desktop_command(user_input: str) -> DesktopCommandPlan | str | None:
    text = (user_input or "").strip()
    normalized = _normalize_text(text)
    if not normalized:
        return None

    for pattern in _SEARCH_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        browser = resolve_desktop_app(match.group("browser"), browsers_only=True)
        if browser is None:
            continue
        query = _strip_search_suffix(match.group("query"))
        if not query:
            return "Tell me what to search for."
        return _build_search_plan(browser, query)

    for pattern in _PROJECT_FOLDER_PATTERNS:
        match = pattern.match(text)
        if match and _is_project_folder_request(match.group("location")):
            return _build_project_folder_plan()

    if normalized in {"project folder", "jarvis project folder", "jarvis folder", "project directory"}:
        return _build_project_folder_plan()

    if normalized.startswith("open any app") or normalized.startswith("open an app"):
        return _build_launch_plan(_DEFAULT_APP)

    open_match = _OPEN_PATTERN.match(text)
    if not open_match:
        return None

    app = resolve_desktop_app(open_match.group("app"))
    if app is None:
        return f"I can open: {_supported_apps_text()}."
    return _build_launch_plan(app)


async def handle_desktop_command(user_input: str) -> str | None:
    planned = plan_desktop_command(user_input)
    if planned is None:
        return None
    if isinstance(planned, str):
        return planned

    primary = await async_launch_application(planned.primary_target, list(planned.primary_args) or None)
    if primary.success:
        return planned.success_message

    if planned.fallback_target:
        fallback = await async_launch_application(planned.fallback_target, list(planned.fallback_args) or None)
        if fallback.success:
            return planned.success_message
        error = fallback.error or primary.error
    else:
        error = primary.error

    app_label = planned.app_label
    detail = f": {error}" if error else "."
    return f"I couldn't open {app_label}{detail}"


__all__ = [
    "DesktopApp",
    "DesktopCommandPlan",
    "handle_desktop_command",
    "plan_desktop_command",
    "resolve_desktop_app",
]
