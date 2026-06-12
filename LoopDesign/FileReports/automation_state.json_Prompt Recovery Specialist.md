# File Report: automation_state.json
**Role:** Prompt Recovery Specialist

## Analysis
The file `automation_state.json` maintains state information for the system's ingestion or automation processes, preventing re-processing of previously seen artifacts.

### Schemas and API Contracts
- Format: JSON Object.
- Schema:
  - `saved_at` (String): ISO 8601 formatted datetime with timezone offset indicating when the state was last persisted (e.g., `"2026-06-11T12:56:50.555604+00:00"`).
  - `seen_fingerprints` (Array of Strings): A list of 64-character hexadecimal strings (SHA-256 hashes), which act as unique identifiers for data that has already been evaluated or processed.

### Assumptions & Dependencies
- Assumptions:
  - System uses SHA-256 (or a similarly 64-char length hash algorithm) to fingerprint files (likely the screenshots mentioned in `automation_ingest.jsonl`) or extracted text chunks.
  - The automation loop checks this state file or loads it into memory to determine if a new file needs processing.
- No system prompts, templating variables, or LLM directives were found in this file.

### Configuration Variables
- None explicit, though it dictates operational state.
