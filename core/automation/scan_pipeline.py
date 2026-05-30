"""Async scan pipeline primitives for live automation ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

SUMMARY_KEYS = (
    "commands_processed",
    "files_ingested",
    "chunks_ingested",
    "failed_files",
    "skipped_files",
    "scanned_files",
)


def blank_scan_summary() -> dict[str, int]:
    return {key: 0 for key in SUMMARY_KEYS}


@dataclass(frozen=True)
class ScanBatch:
    name: str
    candidates: tuple[Path, ...]
    mark_seen: bool
    process: Callable[[Path], Awaitable[dict[str, int]]]
    on_preexisting: Callable[[Path], None] | None = None
    on_error: Callable[[Path, Exception], None] | None = None


ReadinessCheck = Callable[[Path, bool], tuple[bool, str]]


class ScanPipeline:
    def __init__(self, batches: list[ScanBatch] | tuple[ScanBatch, ...]) -> None:
        self._batches = tuple(batches)

    async def run(self, readiness: ReadinessCheck) -> dict[str, int]:
        summary = blank_scan_summary()
        summary["scanned_files"] = sum(len(batch.candidates) for batch in self._batches)

        for batch in self._batches:
            for path in batch.candidates:
                ready, reason = readiness(path, batch.mark_seen)
                if not ready:
                    summary["skipped_files"] += 1
                    if reason == "preexisting" and batch.on_preexisting is not None:
                        batch.on_preexisting(path)
                    continue

                try:
                    delta = await batch.process(path)
                except Exception as exc:  # noqa: BLE001
                    summary["failed_files"] += 1
                    if batch.on_error is not None:
                        batch.on_error(path, exc)
                    continue

                for key, value in delta.items():
                    if key in summary:
                        summary[key] += int(value)

        return summary


__all__ = [
    "ReadinessCheck",
    "ScanBatch",
    "ScanPipeline",
    "blank_scan_summary",
]
