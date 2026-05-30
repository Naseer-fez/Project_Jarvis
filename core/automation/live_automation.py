"""Always-on local automation for command inbox and RAG ingestion."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.automation.scan_pipeline import ScanBatch, ScanPipeline
from core.automation.scan_rules import ScanRoute, build_scan_routes
from core.runtime.bootstrap import _resolve_path

logger = logging.getLogger(__name__)

CommandHandler = Callable[[str], Awaitable[str]]

_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".ini",
    ".log",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_COMMAND_EXTENSIONS = {".txt", ".md", ".task", ".cmd"}

_DEFAULT_DROP_ROOT = "workspace/jarvis_dropbox"
_DEFAULT_SCREENSHOT_DIR = "outputs/screenshots"
_DEFAULT_RECORDING_DIR = "outputs/screen_recordings"


def _cfg_bool(config: Any, section: str, key: str, fallback: bool) -> bool:
    try:
        return bool(config.getboolean(section, key, fallback=fallback))
    except Exception:
        return fallback


def _cfg_float(config: Any, section: str, key: str, fallback: float) -> float:
    try:
        return float(config.get(section, key, fallback=str(fallback)))
    except Exception:
        return fallback


def _cfg_int(config: Any, section: str, key: str, fallback: int) -> int:
    try:
        return int(config.get(section, key, fallback=str(fallback)))
    except Exception:
        return fallback


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


@dataclass
class AutomationStats:
    started_at: str = ""
    last_scan_at: str = ""
    last_error: str = ""
    scanned_files: int = 0
    ingested_files: int = 0
    ingested_chunks: int = 0
    commands_executed: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    live_screen_updates: int = 0


class LiveAutomationEngine:
    """Poll-based automation engine for command inbox and RAG ingestion."""

    def __init__(
        self,
        *,
        config: Any,
        memory: Any,
        llm: Any | None = None,
        command_handler: CommandHandler | None = None,
        desktop_observer: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.llm = llm
        self.command_handler = command_handler
        self.desktop_observer = desktop_observer
        self.notifier = notifier

        self.enabled = _cfg_bool(config, "automation", "enabled", True)
        self.auto_execute_commands = _cfg_bool(
            config,
            "automation",
            "auto_execute_commands",
            True,
        )
        self.watch_screenshots = _cfg_bool(
            config,
            "automation",
            "watch_screenshots",
            True,
        )
        self.watch_recordings = _cfg_bool(
            config,
            "automation",
            "watch_recordings",
            True,
        )
        self.live_screen_enabled = _cfg_bool(
            config,
            "automation",
            "live_screen_enabled",
            True,
        )
        self.ingest_existing_on_start = _cfg_bool(
            config,
            "automation",
            "ingest_existing_on_start",
            False,
        )

        self.poll_interval_seconds = max(
            0.5,
            _cfg_float(config, "automation", "poll_interval_seconds", 3.0),
        )
        self.min_file_age_seconds = max(
            0.0,
            _cfg_float(config, "automation", "min_file_age_seconds", 2.0),
        )
        self.live_screen_interval_seconds = max(
            5.0,
            _cfg_float(config, "automation", "live_screen_interval_seconds", 20.0),
        )
        self.video_frame_interval_seconds = max(
            0.5,
            _cfg_float(config, "automation", "video_frame_interval_seconds", 2.0),
        )
        self.max_video_samples = max(
            1,
            _cfg_int(config, "automation", "max_video_samples", 20),
        )
        self.max_text_chars_per_item = max(
            500,
            _cfg_int(config, "automation", "max_text_chars_per_item", 12000),
        )
        self.chunk_size_chars = max(
            300,
            _cfg_int(config, "automation", "chunk_size_chars", 1200),
        )
        self.chunk_overlap_chars = max(
            0,
            _cfg_int(config, "automation", "chunk_overlap_chars", 120),
        )
        self.max_seen_fingerprints = max(
            200,
            _cfg_int(config, "automation", "max_seen_fingerprints", 10000),
        )

        raw_drop_root = str(
            config.get("automation", "drop_root", fallback=_DEFAULT_DROP_ROOT)
        ).strip()
        self.drop_root = _resolve_path(raw_drop_root or _DEFAULT_DROP_ROOT)
        self.commands_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "commands_folder",
                    fallback=str(self.drop_root / "commands"),
                )
            )
        )
        self.rag_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "rag_folder",
                    fallback=str(self.drop_root / "rag"),
                )
            )
        )
        self.processed_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "processed_folder",
                    fallback=str(self.drop_root / "processed"),
                )
            )
        )
        self.failed_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "failed_folder",
                    fallback=str(self.drop_root / "failed"),
                )
            )
        )
        self.screenshots_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "screenshots_folder",
                    fallback=_DEFAULT_SCREENSHOT_DIR,
                )
            )
        )
        self.recordings_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "recordings_folder",
                    fallback=_DEFAULT_RECORDING_DIR,
                )
            )
        )
        self.log_file = _resolve_path(
            str(
                config.get(
                    "automation",
                    "ingest_log_file",
                    fallback="runtime/automation_ingest.jsonl",
                )
            )
        )
        self.state_file = _resolve_path(
            str(
                config.get(
                    "automation",
                    "state_file",
                    fallback="runtime/automation_state.json",
                )
            )
        )

        self._running = False
        self._task: asyncio.Task | None = None
        self._startup_ts = time.time()
        self._fingerprints: set[str] = set()
        self._fingerprints_order: list[str] = []
        self._last_live_screen_hash = ""
        self._last_live_screen_at = 0.0
        self._stats = AutomationStats()
        self._load_state()

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Live automation is disabled by config.")
            return
        if self._running:
            return
        self._ensure_directories()
        self._running = True
        self._stats.started_at = _iso_now()
        self._task = asyncio.create_task(
            self._run_loop(),
            name="jarvis-live-automation",
        )
        logger.info("Live automation started.")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._save_state()
        logger.info("Live automation stopped.")

    async def enable(self) -> dict[str, Any]:
        self.enabled = True
        await self.start()
        return self.status()

    async def disable(self) -> dict[str, Any]:
        self.enabled = False
        await self.stop()
        return self.status()

    async def force_scan(self) -> dict[str, Any]:
        self._ensure_directories()
        return await self.scan_once()

    async def scan_once(self) -> dict[str, Any]:
        pipeline = ScanPipeline(self._build_scan_batches())
        summary = await pipeline.run(self._scan_readiness)
        self._apply_scan_summary(summary)
        self._save_state()
        return summary

    def _build_scan_batches(self) -> list[ScanBatch]:
        routes = build_scan_routes(
            commands_dir=self.commands_dir,
            rag_dir=self.rag_dir,
            screenshots_dir=self.screenshots_dir,
            recordings_dir=self.recordings_dir,
            command_extensions=_COMMAND_EXTENSIONS,
            image_extensions=_IMAGE_EXTENSIONS,
            video_extensions=_VIDEO_EXTENSIONS,
            watch_screenshots=self.watch_screenshots,
            watch_recordings=self.watch_recordings,
        )

        batches: list[ScanBatch] = []
        for route in routes:
            candidates = tuple(self._iter_files(route.folder, route.allowed_extensions))
            if route.kind == "command":
                batches.append(
                    self._build_command_scan_batch(route, candidates)
                )
                continue
            batches.append(self._build_ingest_scan_batch(route, candidates))
        return batches

    def _build_command_scan_batch(
        self,
        route: ScanRoute,
        candidates: tuple[Path, ...],
    ) -> ScanBatch:
        async def _process(path: Path) -> dict[str, int]:
            await self._process_command_file(path)
            return {"commands_processed": 1}

        return ScanBatch(
            name=route.name,
            candidates=candidates,
            mark_seen=route.mark_seen,
            process=_process,
            on_preexisting=self._remember_file,
            on_error=lambda path, exc: self._handle_scan_failure(route, path, exc),
        )

    def _build_ingest_scan_batch(
        self,
        route: ScanRoute,
        candidates: tuple[Path, ...],
    ) -> ScanBatch:
        async def _process(path: Path) -> dict[str, int]:
            chunks = await self._ingest_file(
                path,
                source=route.source,
                move_after=route.move_after,
            )
            return {
                "files_ingested": 1,
                "chunks_ingested": chunks,
            }

        return ScanBatch(
            name=route.name,
            candidates=candidates,
            mark_seen=route.mark_seen,
            process=_process,
            on_preexisting=self._remember_file,
            on_error=lambda path, exc: self._handle_scan_failure(route, path, exc),
        )

    def _scan_readiness(self, path: Path, mark_seen: bool) -> tuple[bool, str]:
        return self._file_ready(path, mark_seen=mark_seen)

    def _handle_scan_failure(
        self,
        route: ScanRoute,
        path: Path,
        exc: Exception,
    ) -> None:
        logger.warning("%s failed for %s: %s", route.failure_label, path, exc)
        self._stats.last_error = str(exc)
        if route.move_to_failed:
            self._move_to_failed(path, error=str(exc))

    def _apply_scan_summary(self, summary: dict[str, int]) -> None:
        self._stats.scanned_files += int(summary.get("scanned_files", 0))
        self._stats.ingested_files += int(summary.get("files_ingested", 0))
        self._stats.ingested_chunks += int(summary.get("chunks_ingested", 0))
        self._stats.failed_files += int(summary.get("failed_files", 0))
        self._stats.skipped_files += int(summary.get("skipped_files", 0))
        self._stats.last_scan_at = _iso_now()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self._running,
            "auto_execute_commands": self.auto_execute_commands,
            "drop_root": str(self.drop_root),
            "commands_dir": str(self.commands_dir),
            "rag_dir": str(self.rag_dir),
            "screenshots_dir": str(self.screenshots_dir),
            "recordings_dir": str(self.recordings_dir),
            "processed_dir": str(self.processed_dir),
            "failed_dir": str(self.failed_dir),
            "stats": asdict(self._stats),
        }

    def status_line(self) -> str:
        state = "running" if self._running else "stopped"
        return (
            f"Automation {state} (enabled={self.enabled}) | "
            f"commands={self._stats.commands_executed} | "
            f"ingested_files={self._stats.ingested_files} | "
            f"ingested_chunks={self._stats.ingested_chunks} | "
            f"live_screen_updates={self._stats.live_screen_updates}"
        )

    def search_rag(self, query: str, top_k: int = 5) -> str:
        query = _normalize_text(query)
        if not query:
            return "Provide a query after 'rag search'."

        try:
            recalled = self.memory.recall_all(query, top_k=max(top_k, 10))
        except Exception as exc:  # noqa: BLE001
            return f"RAG search failed: {exc}"

        episodes = recalled.get("episodes", []) if isinstance(recalled, dict) else []
        matches: list[tuple[float, str]] = []
        for item in episodes:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "") or "").lower()
            event = str(item.get("event") or item.get("document") or "")
            if "rag" not in category and "[RAG Source]" not in event:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            matches.append((score, event))

        if not matches:
            return f"No RAG matches found for '{query}'."

        matches.sort(key=lambda row: row[0], reverse=True)
        lines: list[str] = []
        for index, (score, event) in enumerate(matches[: max(1, top_k)], start=1):
            source = self._extract_metadata_value(event, "source")
            content = self._extract_metadata_value(event, "content")
            snippet = _truncate(_normalize_text(content or event), 180)
            if source:
                lines.append(f"{index}. [{source}] {snippet} (score={score:.2f})")
            else:
                lines.append(f"{index}. {snippet} (score={score:.2f})")
        return "RAG matches:\n" + "\n".join(lines)

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.scan_once()
                await self._poll_live_screen()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("Live automation loop error: %s", exc, exc_info=True)
                self._stats.last_error = str(exc)
            await asyncio.sleep(self.poll_interval_seconds)

    async def _process_command_file(self, path: Path) -> None:
        command_text = self._read_text_file(path)
        command_text = self._extract_command(command_text)
        if not command_text:
            raise ValueError("Command file is empty.")

        if not self.auto_execute_commands:
            raise RuntimeError("Auto command execution is disabled by config.")

        if self.command_handler is None:
            raise RuntimeError("No command handler is attached.")

        response = await self.command_handler(command_text)
        self._stats.commands_executed += 1
        self._append_log(
            {
                "timestamp": _iso_now(),
                "type": "command",
                "path": str(path),
                "command": command_text,
                "response": str(response or ""),
            }
        )

        self._store_rag_text(
            source="command_result",
            path=path,
            text=f"Command: {command_text}\nResult: {response}",
        )

        processed_path = self._relocate(path, self.processed_dir / "commands")
        result_file = processed_path.with_suffix(processed_path.suffix + ".result.txt")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            f"Command: {command_text}\n\nResult:\n{response}\n",
            encoding="utf-8",
        )

        if self.notifier is not None:
            notify = getattr(self.notifier, "notify", None)
            if callable(notify):
                notify(f"Jarvis command executed from inbox: {command_text}")

    async def _ingest_file(self, path: Path, *, source: str, move_after: bool) -> int:
        text = await asyncio.to_thread(self._extract_text_payload, path)
        if not text:
            text = f"File ingested with no extractable text: {path.name}"

        chunks = self._store_rag_text(source=source, path=path, text=text)

        self._append_log(
            {
                "timestamp": _iso_now(),
                "type": "rag_ingest",
                "path": str(path),
                "source": source,
                "chars": len(text),
                "chunks": chunks,
            }
        )

        if move_after:
            self._relocate(path, self.processed_dir / "rag")

        return chunks

    def _extract_text_payload(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in _TEXT_EXTENSIONS:
            raw = self._read_text_file(path)
            return _truncate(raw, self.max_text_chars_per_item)
        if suffix in _IMAGE_EXTENSIONS:
            text = self._extract_text_from_image(path)
            return _truncate(text, self.max_text_chars_per_item)
        if suffix in _VIDEO_EXTENSIONS:
            text = self._extract_text_from_video(path)
            return _truncate(text, self.max_text_chars_per_item)
        return f"Unsupported file type for direct parsing: {path.name}"

    def _extract_text_from_image(self, path: Path) -> str:
        try:
            from PIL import Image
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return f"OCR dependency missing for image '{path.name}': {exc}"

        try:
            with Image.open(path) as image:
                raw = pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001
            return f"Image OCR failed for '{path.name}': {exc}"

        text = _normalize_text(raw)
        if not text:
            return f"No OCR text found in image '{path.name}'."
        return text

    def _extract_text_from_video(self, path: Path) -> str:
        try:
            import cv2  # type: ignore[import]
            from PIL import Image
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return f"Video OCR dependency missing for '{path.name}': {exc}"

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return f"Could not open video '{path.name}'."

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 1.0
        sample_every_frames = max(1, int(round(fps * self.video_frame_interval_seconds)))

        frame_index = 0
        captured = 0
        snippets: list[str] = []
        try:
            while captured < self.max_video_samples:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % sample_every_frames != 0:
                    frame_index += 1
                    continue
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    raw = pytesseract.image_to_string(image)
                    text = _normalize_text(raw)
                    if text:
                        second = frame_index / max(fps, 1.0)
                        snippets.append(f"[t={second:.1f}s] {text}")
                        captured += 1
                except Exception:
                    pass
                frame_index += 1
        finally:
            cap.release()

        if not snippets:
            return f"No OCR text found in video '{path.name}'."
        return "\n".join(snippets)

    async def _poll_live_screen(self) -> None:
        if not self.live_screen_enabled:
            return
        if self.desktop_observer is None:
            return
        now = time.time()
        if now - self._last_live_screen_at < self.live_screen_interval_seconds:
            return
        self._last_live_screen_at = now

        observe = getattr(self.desktop_observer, "observe", None)
        if not callable(observe):
            return
        try:
            observation = await observe(label="live_automation")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Live screen observation failed: %s", exc)
            return

        ocr_text = _normalize_text(str(getattr(observation, "ocr_text", "") or ""))
        if not ocr_text:
            return

        digest = hashlib.sha256(ocr_text.encode("utf-8", errors="replace")).hexdigest()
        if digest == self._last_live_screen_hash:
            return
        self._last_live_screen_hash = digest

        screenshot_path = str(getattr(observation, "screenshot_path", "") or "")
        text = f"Live screen OCR: {ocr_text}"
        if screenshot_path:
            text += f"\nScreenshot: {screenshot_path}"

        self._store_rag_text(
            source="live_screen",
            path=Path(screenshot_path) if screenshot_path else self.screenshots_dir,
            text=text,
        )
        self._stats.live_screen_updates += 1

    def _store_rag_text(self, *, source: str, path: Path, text: str) -> int:
        clean = str(text or "").strip()
        if not clean:
            return 0

        chunks = self._chunk_text(clean)
        total = len(chunks)
        stored = 0
        for index, chunk in enumerate(chunks, start=1):
            payload = (
                "[RAG Source]\n"
                f"source={source}\n"
                f"path={path}\n"
                f"chunk={index}/{total}\n"
                f"content={chunk}"
            )
            try:
                self.memory.store_episode(payload, category="rag")
                stored += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to store RAG chunk: %s", exc)
                self._stats.last_error = str(exc)
                break
        return stored

    def _chunk_text(self, text: str) -> list[str]:
        size = self.chunk_size_chars
        overlap = min(self.chunk_overlap_chars, max(0, size - 20))
        if len(text) <= size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _file_ready(self, path: Path, *, mark_seen: bool) -> tuple[bool, str]:
        if not path.exists() or not path.is_file():
            return False, "not_file"
        if path.name.startswith("."):
            return False, "hidden"

        stat = path.stat()
        age = time.time() - float(stat.st_mtime)
        if age < self.min_file_age_seconds:
            return False, "too_new"

        if (not self.ingest_existing_on_start) and (stat.st_mtime < self._startup_ts):
            return False, "preexisting"

        if mark_seen:
            fingerprint = self._fingerprint(path, stat)
            if fingerprint in self._fingerprints:
                return False, "seen"
            self._remember_fingerprint(fingerprint)
        return True, "ready"

    def _iter_files(self, folder: Path, allowed_extensions: set[str] | None) -> list[Path]:
        if not folder.exists() or not folder.is_dir():
            return []
        files: list[Path] = []
        for item in folder.iterdir():
            if not item.is_file():
                continue
            if allowed_extensions is not None and item.suffix.lower() not in allowed_extensions:
                continue
            files.append(item)
        files.sort(key=lambda p: p.stat().st_mtime)
        return files

    @staticmethod
    def _extract_command(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""
        first_line = text.splitlines()[0].strip()
        lowered = first_line.lower()
        prefixes = ("command:", "cmd:", "task:")
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix) :].strip()
        return text

    @staticmethod
    def _read_text_file(path: Path, max_bytes: int = 2_000_000) -> str:
        data = path.read_bytes()[: max(1, max_bytes)]
        return data.decode("utf-8", errors="replace")

    def _move_to_failed(self, path: Path, *, error: str) -> None:
        destination = self._relocate(path, self.failed_dir)
        error_file = destination.with_suffix(destination.suffix + ".error.txt")
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text(error + "\n", encoding="utf-8")

    def _relocate(self, source: Path, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = self._unique_path(destination_dir / source.name)
        shutil.move(str(source), str(destination))
        return destination

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _fingerprint(self, path: Path, stat: Any | None = None) -> str:
        if stat is None:
            stat = path.stat()
        raw = f"{path.resolve()}::{int(stat.st_mtime)}::{int(stat.st_size)}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _remember_file(self, path: Path) -> None:
        try:
            self._remember_fingerprint(self._fingerprint(path))
        except Exception:
            return

    def _remember_fingerprint(self, fingerprint: str) -> None:
        if fingerprint in self._fingerprints:
            return
        self._fingerprints.add(fingerprint)
        self._fingerprints_order.append(fingerprint)
        while len(self._fingerprints_order) > self.max_seen_fingerprints:
            oldest = self._fingerprints_order.pop(0)
            self._fingerprints.discard(oldest)

    def _ensure_directories(self) -> None:
        for folder in (
            self.drop_root,
            self.commands_dir,
            self.rag_dir,
            self.processed_dir,
            self.failed_dir,
            self.screenshots_dir,
            self.recordings_dir,
            self.log_file.parent,
            self.state_file.parent,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def _append_log(self, payload: dict[str, Any]) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            seen = raw.get("seen_fingerprints", [])
            if isinstance(seen, list):
                for item in seen[-self.max_seen_fingerprints :]:
                    if isinstance(item, str):
                        self._remember_fingerprint(item)
            stats = raw.get("stats", {})
            if isinstance(stats, dict):
                self._stats = AutomationStats(**{**asdict(self._stats), **stats})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not load automation state: %s", exc)

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": _iso_now(),
                "seen_fingerprints": self._fingerprints_order[-self.max_seen_fingerprints :],
                "stats": asdict(self._stats),
            }
            self.state_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not persist automation state: %s", exc)

    @staticmethod
    def _extract_metadata_value(block: str, key: str) -> str:
        pattern = rf"^{re.escape(key)}=(.*)$"
        match = re.search(pattern, str(block or ""), flags=re.MULTILINE)
        if not match:
            return ""
        return str(match.group(1) or "").strip()


__all__ = ["LiveAutomationEngine", "AutomationStats"]
