"""Routing rules for live automation scan targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ScanRouteKind = Literal["command", "ingest"]


@dataclass(frozen=True)
class ScanRoute:
    name: str
    kind: ScanRouteKind
    folder: Path
    allowed_extensions: set[str] | None
    mark_seen: bool
    source: str = ""
    move_after: bool = False
    move_to_failed: bool = False
    failure_label: str = "Ingestion"


def build_scan_routes(
    *,
    commands_dir: Path,
    rag_dir: Path,
    screenshots_dir: Path,
    recordings_dir: Path,
    command_extensions: set[str],
    image_extensions: set[str],
    video_extensions: set[str],
    watch_screenshots: bool,
    watch_recordings: bool,
) -> tuple[ScanRoute, ...]:
    routes: list[ScanRoute] = [
        ScanRoute(
            name="commands",
            kind="command",
            folder=commands_dir,
            allowed_extensions=command_extensions,
            mark_seen=False,
            move_to_failed=True,
            failure_label="Command ingestion",
        ),
        ScanRoute(
            name="rag",
            kind="ingest",
            folder=rag_dir,
            allowed_extensions=None,
            mark_seen=False,
            source="drop_rag",
            move_after=True,
            move_to_failed=True,
            failure_label="RAG ingestion",
        ),
    ]

    if watch_screenshots:
        routes.append(
            ScanRoute(
                name="screenshots",
                kind="ingest",
                folder=screenshots_dir,
                allowed_extensions=image_extensions,
                mark_seen=True,
                source="screenshot",
                move_after=False,
                move_to_failed=False,
                failure_label="Screenshot ingestion",
            )
        )

    if watch_recordings:
        routes.append(
            ScanRoute(
                name="recordings",
                kind="ingest",
                folder=recordings_dir,
                allowed_extensions=video_extensions,
                mark_seen=True,
                source="recording",
                move_after=False,
                move_to_failed=False,
                failure_label="Recording ingestion",
            )
        )

    return tuple(routes)


__all__ = ["ScanRoute", "ScanRouteKind", "build_scan_routes"]
