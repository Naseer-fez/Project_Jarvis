# Analysis Report for scan_pipeline.py

## Dependencies
- __future__.annotations
- dataclasses.dataclass
- pathlib.Path
- typing.Awaitable
- typing.Callable

## Schemas
- ScanBatch
- ScanBatch attribute: name
- ScanBatch attribute: candidates
- ScanBatch attribute: mark_seen
- ScanBatch attribute: process
- ScanBatch attribute: on_preexisting
- ScanBatch attribute: on_error
- ScanPipeline

## API Contracts
- blank_scan_summary()
- ScanPipeline.__init__(self, batches)

## Configuration Variables
- SUMMARY_KEYS

## Assumptions & Notes
- Module Docstring: Async scan pipeline primitives for live automation ingestion.

