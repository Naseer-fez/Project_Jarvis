# Folder Analysis: audit

## Folder Purpose
Contains components related to audit.

## Findings
- **AUDIT-001** (Critical): The `_ASSIGNMENT_PATTERNS` regex fails to match common secret variables prefixed with other words or underscores (e.g., `db_password`, `user_token`) due to the `\b` word boundary anchor.
- **AUDIT-002** (High): The `_ASSIGNMENT_PATTERNS` regex strictly expects an equals sign (`=`) separator, failing to redact secrets embedded in JSON payloads or Python dictionaries where colons (`:`) are used.
- **AUDIT-003** (Medium): The `_LONG_SECRET` regex fails to redact base64-encoded secrets or complex tokens containing non-alphanumeric characters (such as `+`, `/`, `=`, `-`, `_`).
- **AUDIT-004** (Medium): The `scrub_secrets` function incorrectly silences valid falsy inputs (such as the integer `0`, `0.0`, or boolean `False`), blindly converting them to empty strings.
- **AUDIT-005** (Low): The configuration file contains malformed INI syntax and invalid configuration types.
- **AUDIT-006** (Low): The file intended to act as an SQLite database (`memory.db`) is corrupted or invalid, measuring only 36 bytes with a text payload rather than a valid SQLite header.
- **AUDIT-007** (Low): The `sqlite_file` path is specified as a relative file path, making resolution fragile.

## Risks & Dependencies
See full project roadmap.
