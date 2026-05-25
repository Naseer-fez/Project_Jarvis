from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.automation.live_automation import LiveAutomationEngine


@dataclass
class _Obs:
    ocr_text: str
    screenshot_path: str = ""


class _FakeObserver:
    async def observe(self, label: str = "") -> _Obs:
        _ = label
        return _Obs(ocr_text="Invoice total due: 1240", screenshot_path="outputs/screenshots/live.png")


class _FakeMemory:
    def __init__(self) -> None:
        self.episodes: list[dict] = []

    def store_episode(self, event: str, category: str = "general") -> bool:
        self.episodes.append({"event": event, "category": category})
        return True

    def recall_all(self, query: str, top_k: int = 5) -> dict:
        lowered = query.lower()
        matches = []
        for item in self.episodes:
            event = str(item.get("event", ""))
            if lowered in event.lower():
                matches.append(
                    {
                        "event": event,
                        "category": item.get("category", ""),
                        "score": 0.9,
                    }
                )
        return {
            "preferences": [],
            "episodes": matches[:top_k],
            "conversations": [],
        }


def _make_config(tmp_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    root = tmp_path / "jarvis_dropbox"
    cfg["automation"] = {
        "enabled": "true",
        "auto_execute_commands": "true",
        "drop_root": str(root),
        "commands_folder": str(root / "commands"),
        "rag_folder": str(root / "rag"),
        "processed_folder": str(root / "processed"),
        "failed_folder": str(root / "failed"),
        "watch_screenshots": "false",
        "watch_recordings": "false",
        "live_screen_enabled": "false",
        "ingest_existing_on_start": "true",
        "min_file_age_seconds": "0",
        "poll_interval_seconds": "1",
        "state_file": str(tmp_path / "runtime" / "automation_state.json"),
        "ingest_log_file": str(tmp_path / "runtime" / "automation_ingest.jsonl"),
    }
    return cfg


@pytest.mark.asyncio
async def test_live_automation_processes_command_and_rag_file(tmp_path: Path):
    cfg = _make_config(tmp_path)
    memory = _FakeMemory()

    async def command_handler(command: str) -> str:
        return f"executed: {command}"

    engine = LiveAutomationEngine(
        config=cfg,
        memory=memory,
        command_handler=command_handler,
    )
    engine._ensure_directories()

    command_file = engine.commands_dir / "task_1.txt"
    command_file.write_text("command: remember I like espresso", encoding="utf-8")
    rag_file = engine.rag_dir / "notes.txt"
    rag_file.write_text("Jarvis should track sprint blockers and action items.", encoding="utf-8")

    summary = await engine.force_scan()

    assert summary["commands_processed"] == 1
    assert summary["files_ingested"] >= 1
    assert summary["failed_files"] == 0

    processed_commands = list((engine.processed_dir / "commands").glob("task_1*.txt"))
    processed_rag = list((engine.processed_dir / "rag").glob("notes*.txt"))
    result_files = list((engine.processed_dir / "commands").glob("task_1*.result.txt"))

    assert processed_commands
    assert processed_rag
    assert result_files
    assert any("espresso" in str(item.get("event", "")).lower() for item in memory.episodes)

    rag_response = engine.search_rag("espresso")
    assert "RAG matches" in rag_response


@pytest.mark.asyncio
async def test_live_automation_polls_live_screen_into_rag(tmp_path: Path):
    cfg = _make_config(tmp_path)
    cfg["automation"]["live_screen_enabled"] = "true"
    cfg["automation"]["watch_screenshots"] = "false"
    memory = _FakeMemory()

    engine = LiveAutomationEngine(
        config=cfg,
        memory=memory,
        desktop_observer=_FakeObserver(),
    )

    await engine._poll_live_screen()

    assert any("live screen ocr" in str(item.get("event", "")).lower() for item in memory.episodes)
