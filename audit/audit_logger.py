"""Small security-focused helpers for audit payload scrubbing."""

from __future__ import annotations

import re
import warnings

warnings.warn(
    "The audit.audit_logger module is deprecated. Please use core.logging.logger instead.",
    DeprecationWarning,
    stacklevel=2
)


_ASSIGNMENT_PATTERNS = [
    re.compile(r"(?i)([a-zA-Z0-9_]*(?:password|passwd|token|api[_-]?key))(\s*[:=]\s*)(?:\"[^\"]+\"|'[^']+'|[^\s,;}\]]+)"),
]
_LONG_SECRET = re.compile(r"(?<![a-zA-Z0-9+/=_-])[a-zA-Z0-9+/=_-]{32,}(?![a-zA-Z0-9+/=_-])")


def scrub_secrets(text: str) -> str:
    if text is None:
        return ""
    value = str(text)
    for pattern in _ASSIGNMENT_PATTERNS:
        value = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    value = _LONG_SECRET.sub("[REDACTED]", value)
    return value
