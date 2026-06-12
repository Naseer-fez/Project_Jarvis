ISSUE ID: AUDIT-001
SEVERITY: Critical
CATEGORY: Security vulnerabilities
FILES: d:\AI\Jarvis\audit\audit_logger.py
DESCRIPTION: The `_ASSIGNMENT_PATTERNS` regex fails to match common secret variables prefixed with other words or underscores (e.g., `db_password`, `user_token`) due to the `\b` word boundary anchor.
ROOT CAUSE: The `\b` anchor asserts that the character preceding the keyword must be a non-word character. Since underscores and letters are word characters (`\w`), any prefixed variable name fails to match.
EVIDENCE: `re.compile(r"(?i)\b(password|passwd|token|api[_-]?key)\s*=\s*([^\s,;]+)")`
POTENTIAL IMPACT: Highly sensitive credentials like `db_password` or `access_token` will remain unredacted in the audit logs, leading to immediate credential exposure.
RECOMMENDED FIX: Remove the `\b` anchor or use a non-capturing group to match optional prefixes, such as `(?i)\b[a-zA-Z0-9_]*(password|passwd|token|api[_-]?key)\s*=\s*([^\s,;]+)`.

ISSUE ID: AUDIT-002
SEVERITY: High
CATEGORY: Security vulnerabilities
FILES: d:\AI\Jarvis\audit\audit_logger.py
DESCRIPTION: The `_ASSIGNMENT_PATTERNS` regex strictly expects an equals sign (`=`) separator, failing to redact secrets embedded in JSON payloads or Python dictionaries where colons (`:`) are used.
ROOT CAUSE: The assignment regex `\s*=\s*` hardcodes the equals sign as the only valid key-value separator.
EVIDENCE: `re.compile(r"(?i)\b(password|passwd|token|api[_-]?key)\s*=\s*([^\s,;]+)")`
POTENTIAL IMPACT: Applications logging JSON requests, responses, or dictionary objects will leak secret values in plaintext.
RECOMMENDED FIX: Expand the separator regex to match both equals signs and colons, handling optional formatting, e.g., `\s*[:=]\s*`.

ISSUE ID: AUDIT-003
SEVERITY: Medium
CATEGORY: Security vulnerabilities
FILES: d:\AI\Jarvis\audit\audit_logger.py
DESCRIPTION: The `_LONG_SECRET` regex fails to redact base64-encoded secrets or complex tokens containing non-alphanumeric characters (such as `+`, `/`, `=`, `-`, `_`).
ROOT CAUSE: The character class `[a-zA-Z0-9]` strictly restricts matches to alphanumeric characters, breaking the match continuity when special token characters appear.
EVIDENCE: `_LONG_SECRET = re.compile(r"\b[a-zA-Z0-9]{32,}\b")`
POTENTIAL IMPACT: Secure tokens that utilize standard base64 or URL-safe encoding will bypass redaction entirely if they lack a continuous 32-character alphanumeric chunk.
RECOMMENDED FIX: Update the regex character class to include common token characters: `\b[a-zA-Z0-9\+\/\-_=]{32,}\b`.

ISSUE ID: AUDIT-004
SEVERITY: Medium
CATEGORY: Logic errors
FILES: d:\AI\Jarvis\audit\audit_logger.py
DESCRIPTION: The `scrub_secrets` function incorrectly silences valid falsy inputs (such as the integer `0`, `0.0`, or boolean `False`), blindly converting them to empty strings.
ROOT CAUSE: The expression `text or ""` evaluates to an empty string for all falsy values, rather than just handling `None`.
EVIDENCE: `value = str(text or "")`
POTENTIAL IMPACT: Audit logs will unexpectedly drop valid data, leading to incomplete or inaccurate audit trails when logging payloads that contain zero values or boolean flags.
RECOMMENDED FIX: Explicitly check for `None` instead of relying on truthiness: `value = str("" if text is None else text)`.

ISSUE ID: AUDIT-005
SEVERITY: Low
CATEGORY: Configuration problems
FILES: d:\AI\Jarvis\audit\edge_cases\bad_config.ini
DESCRIPTION: The configuration file contains malformed INI syntax and invalid configuration types.
ROOT CAUSE: Line 1 is missing a closing bracket (`]`) for the section header, and line 2 assigns a non-numeric string (`invalid`) to a port variable.
EVIDENCE: `[dashboard\nport=invalid`
POTENTIAL IMPACT: If processed, `configparser` will raise a `MissingSectionHeaderError` or a `ValueError`, causing application startup crashes.
RECOMMENDED FIX: Correct the syntax by appending the closing bracket (`[dashboard]`) and setting a valid integer for the port, e.g., `port=8080`.

ISSUE ID: AUDIT-006
SEVERITY: Low
CATEGORY: Database issues
FILES: d:\AI\Jarvis\audit\edge_cases\memory\memory.db
DESCRIPTION: The file intended to act as an SQLite database (`memory.db`) is corrupted or invalid, measuring only 36 bytes with a text payload rather than a valid SQLite header.
ROOT CAUSE: An SQLite database requires a standard 100-byte header starting with "SQLite format 3\000". A 36-byte file lacks the valid SQLite database structure.
EVIDENCE: The `memory.db` file size is exactly 36 bytes.
POTENTIAL IMPACT: Any SQLite driver attempting to connect to or query this database file will throw a `DatabaseError: file is not a database`, resulting in database connection failures.
RECOMMENDED FIX: Delete the invalid `memory.db` file and allow the application's SQLite driver to automatically initialize a properly formatted database upon startup.

ISSUE ID: AUDIT-007
SEVERITY: Low
CATEGORY: Configuration problems
FILES: d:\AI\Jarvis\audit\edge_cases\test_config.ini
DESCRIPTION: The `sqlite_file` path is specified as a relative file path, making resolution fragile.
ROOT CAUSE: Hardcoding relative paths depends strictly on the application's current working directory matching the project root at runtime.
EVIDENCE: `sqlite_file=audit/edge_cases/memory/memory.db`
POTENTIAL IMPACT: Running the application from any directory other than the root will cause file resolution failures or lead to an unintended database being created in the wrong directory.
RECOMMENDED FIX: Use absolute paths or an environment-variable-based path resolution strategy for the database file target.
