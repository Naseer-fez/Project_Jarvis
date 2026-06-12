# Dependency Analysis: jarvis_memory.db

## Overview
An SQLite database acting as the primary application memory and long-term storage context (Hybrid Memory).

## Schemas / API Contracts
Contains the following tables:
- `facts`: (key, value, source, created_at, updated_at, metadata)
- `preferences`: (key, value, updated_at)
- `episodes`: (id, event, category, timestamp)
- `conversations`: (id, user_input, assistant_response, session_id, timestamp)
- `actions`: (id, action, result, success, metadata, timestamp)

## Assumptions & Dependencies
- Assumes `sqlite3`.
- Assumes integration with `tesseract` for vision/OCR workflows. (Derived from `episodes` table: `Image OCR failed for '...png': tesseract is not installed or it's not in your PATH.`).
- GUI automation depends on config variables (`GUI automation is disabled by config`).
- Network configurations: Relies on an external/internal service port that actively refused connection (`[WinError 10061]`), pointing to a local inference or utility server that must be running (used for `screen_understand`).
- Assumes the user goes by the name "Bob" (hardcoded or learned preference seen in conversation logs).

## Prompts & Conversational Constructs
- Extracted system prompt / capability list from DB memory: The assistant identifies as "Jarvis" and provides a bulleted list of 10 supported tasks: Schedule management, Email management, Reminders, Weather updates, News updates, Internet search, Music control, Smart home control, Note-taking, Task automation. (Extracted to Prompts folder).
