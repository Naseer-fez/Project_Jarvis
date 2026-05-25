"""Reusable request-classification rules for controller routing."""

from __future__ import annotations

DESKTOP_CONTROL_KEYWORDS = (
    "mouse",
    "cursor",
    "desktop",
    "screen",
    "keyboard",
    "hotkey",
    "click",
    "scroll",
    "drag",
    "clipboard",
)

AGENTIC_KEYWORDS = (
    "search",
    "look up",
    "find",
    "check",
    "scrape",
    "get",
    "download",
    "fetch",
    "read",
    "analyze",
    "create",
    "make",
    "write",
    "run",
    "execute",
    "automate",
    "browse",
    "internet",
    "online",
    "web",
    "website",
    "latest",
    "current",
    "today",
    "live",
    "news",
    "price",
    "weather",
    "stats",
    "score",
    "runs",
    "toss",
    "ipl",
    "match",
    "mouse",
    "cursor",
    "desktop",
    "screen",
    "keyboard",
    "hotkey",
    "click",
    "scroll",
    "drag",
    "clipboard",
)

# Phrases that unambiguously indicate the user wants a live web search.
# When any of these are detected Jarvis skips the LLM planner entirely
# and calls web_search directly, then synthesises the answer.
WEB_SEARCH_EXPLICIT_PHRASES = (
    "search the web",
    "search the internet",
    "search online",
    "browse the web",
    "browse the internet",
    "browse online",
    "look it up online",
    "look up online",
    "google it",
    "google for",
    "find online",
    "find on the internet",
    "find on the web",
    "search for",
    "search web for",
    "web search",
    "internet search",
)

LIVE_WEB_HINTS = (
    "internet",
    "online",
    "web",
    "website",
    "latest",
    "current",
    "today",
    "live",
    "news",
    "price",
    "weather",
    "score",
    "stats",
    "runs",
    "toss",
    "ipl",
    "match",
)

LIVE_WEB_REQUEST_MARKERS = (
    "search",
    "browse",
    "find",
    "check",
    "look up",
    "google",
    "get",
    "give me",
    "tell me",
    "update",
    "use internet",
    "what is",
    "who is",
    "when is",
)

ACTIVE_WINDOW_PHRASES = (
    "active app",
    "active window",
    "focused window",
    "foreground window",
    "foreground app",
    "which app is active",
    "what app is active",
    "what window is active",
    "which window is active",
)


def looks_like_desktop_control_request(lowered: str) -> bool:
    return any(keyword in lowered for keyword in DESKTOP_CONTROL_KEYWORDS)


def is_explicit_web_search(lowered: str) -> bool:
    """Return True when the user unambiguously asks for a live web search."""
    return any(phrase in lowered for phrase in WEB_SEARCH_EXPLICIT_PHRASES)


def should_force_web_search(lowered: str) -> bool:
    if is_explicit_web_search(lowered):
        return True
    if not any(hint in lowered for hint in LIVE_WEB_HINTS):
        return False
    return any(marker in lowered for marker in LIVE_WEB_REQUEST_MARKERS)


def is_active_window_request(lowered: str) -> bool:
    if any(phrase in lowered for phrase in ACTIVE_WINDOW_PHRASES):
        return True
    if "watch the screen" in lowered and "app" in lowered:
        return True
    if "screen" in lowered and "active" in lowered and (
        "app" in lowered or "window" in lowered
    ):
        return True
    return False


__all__ = [
    "ACTIVE_WINDOW_PHRASES",
    "AGENTIC_KEYWORDS",
    "DESKTOP_CONTROL_KEYWORDS",
    "LIVE_WEB_HINTS",
    "LIVE_WEB_REQUEST_MARKERS",
    "WEB_SEARCH_EXPLICIT_PHRASES",
    "is_active_window_request",
    "is_explicit_web_search",
    "looks_like_desktop_control_request",
    "should_force_web_search",
]
