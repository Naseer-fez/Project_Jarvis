"""Reusable screen observation and before/after change detection."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

from core.desktop.contracts import DesktopChange, DesktopObservation, ScreenTarget


ObservationHandler = Callable[..., Any]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _result_success(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(getattr(result, "success", False))


def _result_error(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("error", "") or "")
    return str(getattr(result, "error", "") or "")


def _result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, dict):
            return data
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return {}

    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _hash_path(path_value: str) -> str:
    if not path_value:
        return ""
    try:
        path = Path(path_value)
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _target_from_payload(payload: dict[str, Any]) -> ScreenTarget | None:
    try:
        label = str(payload.get("text") or payload.get("label") or "")
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        width = int(payload.get("w", payload.get("width", 0)))
        height = int(payload.get("h", payload.get("height", 0)))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not label and width <= 0 and height <= 0:
        return None
    return ScreenTarget(
        label=label,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=confidence,
        metadata={k: v for k, v in payload.items() if k not in {"text", "label", "x", "y", "w", "h", "width", "height", "confidence"}},
    )


class DesktopObserver:
    """Capture normalized evidence about the current desktop state."""

    def __init__(
        self,
        *,
        capture_screen: ObservationHandler | None = None,
        active_window: ObservationHandler | None = None,
        ocr: ObservationHandler | None = None,
    ) -> None:
        self._capture_screen = capture_screen
        self._active_window = active_window
        self._ocr = ocr

    async def observe(self, label: str = "") -> DesktopObservation:
        screenshot_path = ""
        screenshot_fingerprint = ""
        active_window: dict[str, Any] = {}
        ocr_text = ""
        targets: list[ScreenTarget] = []
        metadata: dict[str, Any] = {"label": label} if label else {}
        errors: list[str] = []
        confidence = 0.0

        capture = self._capture_screen or self._default_capture_screen
        try:
            result = await _maybe_await(capture())
            if _result_success(result):
                payload = _result_payload(result)
                screenshot_path = str(payload.get("path", "") or "")
                screenshot_fingerprint = str(
                    payload.get("fingerprint", "") or _hash_path(screenshot_path)
                )
                metadata["screenshot"] = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"path", "fingerprint"}
                }
                confidence += 0.45
            else:
                errors.append(_result_error(result) or "screenshot unavailable")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"screenshot failed: {exc}")

        active = self._active_window or self._default_active_window
        try:
            result = await _maybe_await(active())
            if _result_success(result):
                active_window = _result_payload(result)
                confidence += 0.3
            else:
                errors.append(_result_error(result) or "active window unavailable")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"active window failed: {exc}")

        ocr = self._ocr or self._default_ocr
        try:
            result = await _maybe_await(ocr())
            if _result_success(result):
                payload = _result_payload(result)
                ocr_text = str(
                    payload.get("ocr_text")
                    or payload.get("description")
                    or payload.get("text")
                    or ""
                )
                raw_targets = payload.get("matches") or payload.get("lines") or []
                for match in raw_targets:
                    if isinstance(match, dict):
                        target = _target_from_payload(match)
                        if target is not None:
                            targets.append(target)
                if ocr_text or targets:
                    confidence += 0.2
            else:
                errors.append(_result_error(result) or "ocr unavailable")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ocr failed: {exc}")

        confidence = min(1.0, confidence)
        low_confidence_reason = ""
        if confidence < 0.5:
            low_confidence_reason = "; ".join(error for error in errors if error) or "not enough desktop evidence"
        if errors:
            metadata["observation_errors"] = errors

        return DesktopObservation(
            screenshot_path=screenshot_path,
            screenshot_fingerprint=screenshot_fingerprint,
            active_window=active_window,
            ocr_text=ocr_text,
            targets=targets,
            confidence=confidence,
            low_confidence_reason=low_confidence_reason,
            metadata=metadata,
        )

    def compare(
        self,
        before: DesktopObservation | None,
        after: DesktopObservation | None,
    ) -> DesktopChange:
        if before is None or after is None:
            return DesktopChange(
                changed=False,
                confidence=0.0,
                summary="Missing before or after observation.",
                before_observation_id=getattr(before, "observation_id", ""),
                after_observation_id=getattr(after, "observation_id", ""),
            )

        changed_signals: list[str] = []
        metadata: dict[str, Any] = {}

        if before.screenshot_fingerprint and after.screenshot_fingerprint:
            metadata["screenshot_fingerprint_before"] = before.screenshot_fingerprint
            metadata["screenshot_fingerprint_after"] = after.screenshot_fingerprint
            if before.screenshot_fingerprint != after.screenshot_fingerprint:
                changed_signals.append("screenshot changed")

        before_title = str(before.active_window.get("title", "") or "")
        after_title = str(after.active_window.get("title", "") or "")
        if before_title or after_title:
            metadata["active_window_before"] = before_title
            metadata["active_window_after"] = after_title
            if before_title != after_title:
                changed_signals.append("active window changed")

        if before.ocr_text or after.ocr_text:
            metadata["ocr_before_length"] = len(before.ocr_text)
            metadata["ocr_after_length"] = len(after.ocr_text)
            if before.ocr_text != after.ocr_text:
                changed_signals.append("visible text changed")

        changed = bool(changed_signals)
        if changed:
            confidence = max(before.confidence, after.confidence, 0.7)
            summary = "; ".join(changed_signals)
        else:
            confidence = min(before.confidence, after.confidence)
            summary = "No observable desktop change detected."

        return DesktopChange(
            changed=changed,
            confidence=confidence,
            summary=summary,
            before_observation_id=before.observation_id,
            after_observation_id=after.observation_id,
            metadata=metadata,
        )

    @staticmethod
    def _default_capture_screen() -> Any:
        from core.tools.screen import capture_screen

        return capture_screen()

    @staticmethod
    def _default_active_window() -> Any:
        from core.tools.gui_control import get_active_window

        return get_active_window()

    @staticmethod
    def _default_ocr() -> Any:
        from core.tools.screen import read_screen_text

        return read_screen_text(include_lines=True)


__all__ = ["DesktopObserver"]
