# Folder Analysis: integrations

## Folder Purpose
Contains components related to integrations.

## Findings
- **INTEGRATIONS-001** (High): The `take_screenshot` tool accepts a user-provided `path` and passes it directly to `os.path.abspath` without verifying that the resolved path is within a safe, intended directory. This permits an arbitrary file write (path traversal vulnerability) anywhere on the system.
- **INTEGRATIONS-002** (High): The `_add_event` method executes within an asynchronous thread pool but performs read-modify-write operations on `CALENDAR_PATH` without locks, exposing it to race conditions. Additionally, it uses naive string replacement `.replace("END:VCALENDAR", block + "END:VCALENDAR")` which will aggressively replace all occurrences, including any `END:VCALENDAR` strings maliciously or accidentally included in an event title, severely corrupting the file.
- **INTEGRATIONS-003** (High): When parsing the start and end times for creating an event, `_to_rfc3339` unconditionally attaches UTC `timezone.utc` to any naive datetimes. When combined with the intended `timeZone` field in the API payload, Google Calendar interprets the absolute time as UTC and incorrectly shifts it into the provided timezone.
- **INTEGRATIONS-004** (Medium): The `calendar` integration returns `True` unconditionally in its `is_available()` method, but its `_list_events` utility relies heavily on the third-party libraries `icalendar` and `dateutil`. If these libraries are missing, the integration registers successfully but throws runtime exceptions.
- **INTEGRATIONS-005** (Medium): The `IntegrationRegistry.get_tools()` method iterates over all integrated instances and returns their tool schemas without checking the active `self._tool_owner` mapping. If an integration overrides a tool name belonging to another integration, both the overridden and overriding tool schemas are exported.
- **INTEGRATIONS-006** (Low): In `_get_message_meta()`, if the HTTP request to fetch individual message metadata fails (e.g., due to a rate-limiting 429 or 404), the method silently parses the error payload and returns an empty dictionary.

## Risks & Dependencies
See full project roadmap.
