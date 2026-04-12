"""Profile synthesis helpers."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any


def _strip_wrappers(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


class ProfileSynthesizer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def should_run(self, profile: Any) -> bool:
        try:
            count = int(getattr(profile, "interaction_count", 0))
            return count >= 20 and count % 20 == 0
        except Exception:
            return False

    async def synthesize(
        self,
        conversations: list[str],
        profile: Any,
    ) -> dict[str, Any]:
        prompt = (
            "Update the user profile from the conversation snippets below.\n"
            "Return strict JSON only, where each key maps to "
            '{"value": ..., "confidence": 0.0-1.0}.\n\n'
            + "\n".join(conversations[-20:])
        )

        complete = getattr(self.llm, "complete", None)
        if complete is None:
            return {"error": "llm_unavailable", "updated_fields": []}

        try:
            raw = complete(prompt)
            if inspect.isawaitable(raw):
                raw = await raw
            payload = json.loads(_strip_wrappers(str(raw)))
        except json.JSONDecodeError:
            return {"error": "invalid_json", "updated_fields": []}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "updated_fields": []}

        if not isinstance(payload, dict):
            return {"error": "invalid_payload", "updated_fields": []}

        try:
            updated_fields = list(profile.apply_delta(payload))
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "updated_fields": []}

        return {"updated_fields": updated_fields, "delta": payload}


__all__ = ["ProfileSynthesizer"]
