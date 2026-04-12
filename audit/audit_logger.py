"""Small security-focused helpers for audit payload scrubbing."""

from __future__ import annotations

import re


_ASSIGNMENT_PATTERNS = [
    re.compile(r"(?i)\b(password|passwd|token|api[_-]?key)\s*=\s*([^\s,;]+)"),
]
_LONG_SECRET = re.compile(r"\b[a-zA-Z0-9]{32,}\b")


def scrub_secrets(text: str) -> str:
    value = str(text or "")
    for pattern in _ASSIGNMENT_PATTERNS:
        value = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    value = _LONG_SECRET.sub("[REDACTED]", value)
    return value
