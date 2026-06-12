ISSUE ID: INTEGRATIONS-001
SEVERITY: High
CATEGORY: Security vulnerabilities
FILES: `integrations/clients/computer_control.py`
DESCRIPTION: The `take_screenshot` tool accepts a user-provided `path` and passes it directly to `os.path.abspath` without verifying that the resolved path is within a safe, intended directory. This permits an arbitrary file write (path traversal vulnerability) anywhere on the system.
ROOT CAUSE: Lack of directory boundary checking before writing files to the disk.
EVIDENCE: `path = os.path.abspath(str(args.get("path", "outputs/screenshot.png") ...)` followed directly by `await loop.run_in_executor(None, lambda: pyautogui.screenshot(path))` allows paths like `../../../../Windows/System32/somefile.png`.
POTENTIAL IMPACT: Arbitrary file overwrite, potentially replacing critical system files or injecting malicious files.
RECOMMENDED FIX: Validate that the absolute path returned by `os.path.abspath` resolves inside the intended base output directory using `os.path.commonpath`.

ISSUE ID: INTEGRATIONS-002
SEVERITY: High
CATEGORY: Race conditions
FILES: `integrations/clients/calendar.py`
DESCRIPTION: The `_add_event` method executes within an asynchronous thread pool but performs read-modify-write operations on `CALENDAR_PATH` without locks, exposing it to race conditions. Additionally, it uses naive string replacement `.replace("END:VCALENDAR", block + "END:VCALENDAR")` which will aggressively replace all occurrences, including any `END:VCALENDAR` strings maliciously or accidentally included in an event title, severely corrupting the file.
ROOT CAUSE: Missing file locking for concurrent thread pool operations and usage of fragile string manipulation instead of robust ICS object parsing.
EVIDENCE: `content = CALENDAR_PATH.read_text()` followed immediately by `content.replace("END:VCALENDAR", ...)` and `CALENDAR_PATH.write_text(updated)` without `threading.Lock` or `filelock`.
POTENTIAL IMPACT: Complete corruption of the `.ics` calendar file or loss of events when concurrent calls are made.
RECOMMENDED FIX: Synchronize file access using a locking mechanism (e.g., `filelock`). Replace string manipulation with the `icalendar` library to add components to the calendar safely.

ISSUE ID: INTEGRATIONS-003
SEVERITY: High
CATEGORY: Logic errors
FILES: `integrations/clients/google_calendar.py`
DESCRIPTION: When parsing the start and end times for creating an event, `_to_rfc3339` unconditionally attaches UTC `timezone.utc` to any naive datetimes. When combined with the intended `timeZone` field in the API payload, Google Calendar interprets the absolute time as UTC and incorrectly shifts it into the provided timezone.
ROOT CAUSE: `dt.replace(tzinfo=timezone.utc)` blindly overrides the user's intended local time when no offset is specified.
EVIDENCE: `dt.replace(tzinfo=timezone.utc)` followed by `{ "dateTime": self._to_rfc3339(start_str), "timeZone": tz }` in the API body.
POTENTIAL IMPACT: Events get scheduled at incorrect offset hours resulting in entirely wrong meeting times.
RECOMMENDED FIX: Modify `_to_rfc3339` to accept the intended timezone, parse it (e.g. using `zoneinfo`), and apply the target timezone instead of blindly defaulting to UTC.

ISSUE ID: INTEGRATIONS-004
SEVERITY: Medium
CATEGORY: Dependency issues
FILES: `integrations/clients/calendar.py`
DESCRIPTION: The `calendar` integration returns `True` unconditionally in its `is_available()` method, but its `_list_events` utility relies heavily on the third-party libraries `icalendar` and `dateutil`. If these libraries are missing, the integration registers successfully but throws runtime exceptions.
ROOT CAUSE: `is_available()` ignores the dynamic imports made within the class methods.
EVIDENCE: `def is_available(self) -> bool: return True` paired with `from icalendar import Calendar` inside the `_list_events` method.
POTENTIAL IMPACT: Integration is actively exposed to the planner but crashes abruptly with an `ImportError` upon execution.
RECOMMENDED FIX: Wrap the imports of `icalendar` and `dateutil` in a `try...except ImportError:` block directly inside `is_available()` and return `False` upon failure.

ISSUE ID: INTEGRATIONS-005
SEVERITY: Medium
CATEGORY: Architectural inconsistencies
FILES: `integrations/registry.py`
DESCRIPTION: The `IntegrationRegistry.get_tools()` method iterates over all integrated instances and returns their tool schemas without checking the active `self._tool_owner` mapping. If an integration overrides a tool name belonging to another integration, both the overridden and overriding tool schemas are exported.
ROOT CAUSE: The iteration over `self._integrations.values()` blindly appends all schemas without validating tool ownership.
EVIDENCE: `for integration in self._integrations.values(): tools = integration.get_tools() or []` does not verify if the integration actively owns the tool in `self._tool_owner`.
POTENTIAL IMPACT: The system outputs duplicate/stale tool schemas to the planner agent, causing LLM confusion, context payload bloat, and unpredictable routing behavior.
RECOMMENDED FIX: Update `get_tools()` to cross-reference each tool's name with `self._tool_owner` to guarantee it is only included if the current integration holds active ownership.

ISSUE ID: INTEGRATIONS-006
SEVERITY: Low
CATEGORY: Error handling gaps
FILES: `integrations/clients/gmail.py`
DESCRIPTION: In `_get_message_meta()`, if the HTTP request to fetch individual message metadata fails (e.g., due to a rate-limiting 429 or 404), the method silently parses the error payload and returns an empty dictionary.
ROOT CAUSE: The method lacks an explicit check for `resp.status == 200` before accessing payload fields.
EVIDENCE: `data = await resp.json(); header_map = {}; for h in data.get("payload", {}).get("headers", []):` will fail silently because the error response lacks a payload node.
POTENTIAL IMPACT: Users receive empty or blank summaries for unread emails without any indication that a background API error occurred.
RECOMMENDED FIX: Implement `if resp.status != 200: ...` to properly log the failure and either omit the message from the final dataset or report the fetch error to the user.
