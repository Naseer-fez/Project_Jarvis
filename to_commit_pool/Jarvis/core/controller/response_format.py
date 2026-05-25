"""Small text formatting helpers shared by controller flows."""

from __future__ import annotations

import re
from typing import Any


def normalize_inline_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def infer_app_name_from_title(title: str) -> str:
    cleaned = normalize_inline_whitespace(title)
    for prefix in ("Administrator: ", "Administrator - "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    for separator in (" - ", " | ", " â€” ", " â€“ "):
        if separator not in cleaned:
            continue
        parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
        if parts:
            return parts[-1]

    return cleaned


def parse_web_search_output(raw_results: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "query": "",
        "original_request": "",
        "summary": "",
        "sources": [],
    }

    current_source: dict[str, str] | None = None
    for raw_line in str(raw_results or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Search query used: "):
            parsed["query"] = line.split(": ", 1)[1].strip()
            continue
        if line.startswith("Original request: "):
            parsed["original_request"] = line.split(": ", 1)[1].strip()
            continue
        if line.startswith("Summary: "):
            parsed["summary"] = line.split(": ", 1)[1].strip()
            continue
        if re.match(r"^\d+\.\s+", line):
            title = re.sub(r"^\d+\.\s+", "", line).strip()
            current_source = {"title": title, "url": "", "snippet": ""}
            parsed["sources"].append(current_source)
            continue
        if line.startswith("URL: ") and current_source is not None:
            current_source["url"] = line.split(": ", 1)[1].strip()
            continue
        if line.startswith("Snippet: ") and current_source is not None:
            current_source["snippet"] = line.split(": ", 1)[1].strip()

    return parsed


def render_web_search_response(raw_results: str) -> str:
    parsed = parse_web_search_output(raw_results)
    summary = normalize_inline_whitespace(parsed.get("summary", ""))
    sources = parsed.get("sources", []) or []
    source_labels = [
        f"{source['title']} ({source['url']})"
        for source in sources
        if source.get("title") and source.get("url")
    ]

    if summary and source_labels:
        return f"{summary}\nSources: {'; '.join(source_labels[:3])}"
    if summary:
        return summary
    if source_labels:
        return "I found live web results, but I could not safely compress them further.\nSources: " + "; ".join(
            source_labels[:3]
        )
    return raw_results


__all__ = [
    "infer_app_name_from_title",
    "normalize_inline_whitespace",
    "parse_web_search_output",
    "render_web_search_response",
]
