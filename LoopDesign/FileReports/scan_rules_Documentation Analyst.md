# Analysis Report for scan_rules.py

## Dependencies
- __future__.annotations
- dataclasses.dataclass
- pathlib.Path
- typing.Literal

## Schemas
- ScanRoute
- ScanRoute attribute: name
- ScanRoute attribute: kind
- ScanRoute attribute: folder
- ScanRoute attribute: allowed_extensions
- ScanRoute attribute: mark_seen
- ScanRoute attribute: source
- ScanRoute attribute: move_after
- ScanRoute attribute: move_to_failed
- ScanRoute attribute: failure_label

## API Contracts
- build_scan_routes()

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Routing rules for live automation scan targets.

