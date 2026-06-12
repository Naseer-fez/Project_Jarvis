# Analysis Report for live_automation.py

## Dependencies
- __future__.annotations
- asyncio
- contextlib
- hashlib
- json
- logging
- re
- shutil
- time
- dataclasses.asdict
- dataclasses.dataclass
- datetime.datetime
- datetime.timezone
- pathlib.Path
- typing.Any
- typing.Awaitable
- typing.Callable
- core.automation.scan_pipeline.ScanBatch
- core.automation.scan_pipeline.ScanPipeline
- core.automation.scan_rules.ScanRoute
- core.automation.scan_rules.build_scan_routes
- core.runtime.paths._resolve_path

## Schemas
- AutomationStats
- AutomationStats attribute: started_at
- AutomationStats attribute: last_scan_at
- AutomationStats attribute: last_error
- AutomationStats attribute: scanned_files
- AutomationStats attribute: ingested_files
- AutomationStats attribute: ingested_chunks
- AutomationStats attribute: commands_executed
- AutomationStats attribute: failed_files
- AutomationStats attribute: skipped_files
- AutomationStats attribute: live_screen_updates
- LiveAutomationEngine

## API Contracts
- _cfg_bool(config, section, key, fallback)
- _cfg_float(config, section, key, fallback)
- _cfg_int(config, section, key, fallback)
- _iso_now()
- _normalize_text(value)
- _truncate(text, max_chars)
- LiveAutomationEngine.__init__(self)
- LiveAutomationEngine._build_command_scan_batch(self, route, candidates)
- LiveAutomationEngine._build_ingest_scan_batch(self, route, candidates)
- LiveAutomationEngine._scan_readiness(self, path, mark_seen)
- LiveAutomationEngine._handle_scan_failure(self, route, path, exc)
- LiveAutomationEngine._apply_scan_summary(self, summary)
- LiveAutomationEngine.status(self)
- LiveAutomationEngine.status_line(self)
- LiveAutomationEngine._extract_text_payload(self, path)
- LiveAutomationEngine._file_ready(self, path)
- LiveAutomationEngine._extract_command(raw_text)
- LiveAutomationEngine._move_to_failed(self, path)
- LiveAutomationEngine._relocate(self, source, destination_dir)
- LiveAutomationEngine._unique_path(path)
- LiveAutomationEngine._fingerprint(self, path, stat)
- LiveAutomationEngine._remember_file(self, path)
- LiveAutomationEngine._remember_fingerprint(self, fingerprint)
- LiveAutomationEngine._ensure_directories(self)
- LiveAutomationEngine._load_state(self)
- LiveAutomationEngine._save_state(self)
- LiveAutomationEngine._extract_metadata_value(block, key)

## Configuration Variables
- _TEXT_EXTENSIONS
- _IMAGE_EXTENSIONS
- _VIDEO_EXTENSIONS
- _COMMAND_EXTENSIONS
- _DEFAULT_DROP_ROOT
- _DEFAULT_SCREENSHOT_DIR
- _DEFAULT_RECORDING_DIR

## Assumptions & Notes
- Module Docstring: Always-on local automation for command inbox and RAG ingestion.

