# File Report: automation_state.json
Role: Runtime Investigator

## Overview
This file serves as the persistence layer for the automation component's tracking state. It maintains a record of processed items to ensure idempotency and prevent duplicate processing across runtime sessions.

## Schemas & Contracts
The file contains a single JSON object with the following schema:
- `saved_at` (String): ISO-8601 timestamp with timezone indicating when the state was last persisted. Example: `"2026-06-11T12:56:50.555604+00:00"`.
- `seen_fingerprints` (Array of Strings): A list of SHA-256 (or similar 64-character hex) hashes representing unique identifiers for items that have already been ingested or processed by the automation loop. 

## Assumptions & API Contracts
- The automation execution loop loads this state upon initialization to populate a set of already-processed fingerprints.
- During execution, any new item (e.g., a screenshot or ingested document) has its fingerprint computed. If it matches an entry in `seen_fingerprints`, the item is skipped.
- The automation loop periodically (or gracefully upon exit) updates and overwrites `automation_state.json` to append new fingerprints and update `saved_at`.

## Execution Paths & State Transitions
- **Initialization Transition**: `Empty State -> Loaded State` (Reads JSON, parses array into a fast-lookup data structure like a Set).
- **Processing Loop**: For each event -> Compute Hash -> Check against Set.
- **Persistence Transition**: `Active State -> Flushed State` (Writes updated Set back to `automation_state.json`).

## Dependencies & Variables
- Dependencies: The hashing algorithm used to generate the 64-char strings (likely SHA-256).
- Configuration Variables: The location of this state file is hardcoded or configured as `runtime/automation_state.json`.

## Prompts Found
None.
