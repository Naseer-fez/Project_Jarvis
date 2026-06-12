# Complete Feature Catalog

## 1. Web Dashboard & User Interface
- **Purpose**: Provides a comprehensive web interface for users to monitor Jarvis's state, view memory, manage goals, and control automation.
- **Inputs**: HTTP requests, user authentication credentials (username/password), session cookies, `X-Dashboard-Token`. Form inputs for commands and searches.
- **Outputs**: Rendered HTML pages (`/`, `/memory`, `/goals`, `/search`, `/converter`, `/health-ui`, `/clicker`, `/ai-os`), JSON API responses.
- **User actions**: Logging in/out, searching memory, viewing goal statuses, starting/stopping the auto-clicker, sending direct text commands.
- **Business rules**: All protected routes require a valid `jarvis_session` cookie or an `X-Dashboard-Token` header. If the environment is `production` and the token is set to the default insecure token, the server refuses to start.
- **Edge cases**: Expired session cookies, unauthorized access attempts, missing memory database.
- **Dependencies**: FastAPI, Uvicorn, Jinja2, SQLite3.

## 2. Auto-Clicker & GUI Audit System
- **Purpose**: Continuously or periodically interact with screen elements automatically and maintain visual logs of GUI actions.
- **Inputs**: Target string or image to match, interval in seconds, minimum confidence threshold (0.0 to 1.0), and a continuous mode toggle.
- **Outputs**: Automated mouse clicks on the host OS, execution logs, and screenshots saved to `outputs/gui_audit`.
- **User actions**: Start/Stop the clicker via UI, clear logs, view recent screenshots.
- **Business rules**: Exposes a REST API (`/api/clicker/start`, `/api/clicker/stop`). The backend delegates actual clicking to the main event loop to ensure thread-safety with `pyautogui`. Maintains an in-memory queue of the last 200 log messages.
- **Edge cases**: Target not visible on the screen, screen resolution changes, blocked screen elements.
- **Dependencies**: PyAutoGUI, OpenCV (for computer vision matching).

## 3. Live Automation (Command Inbox & RAG Ingestion)
- **Purpose**: An always-on background polling mechanism that ingests files dropped into designated folders and periodically scans the screen.
- **Inputs**: 
  - Text/Task files dropped in `workspace/jarvis_dropbox/commands`.
  - Document/Media files dropped in `workspace/jarvis_dropbox/rag`.
  - Live screen updates (screenshots/OCR).
- **Outputs**: Executed DAG plans or commands, RAG chunk embeddings in the memory DB, relocated files to `processed` or `failed` folders.
- **User actions**: Users drop a file into the inbox folder or enable screen recording features.
- **Business rules**: Polls every `poll_interval_seconds`. Uses exponential backoff on failures. Extracts text from documents/videos/images using a `PayloadExtractor`. Keeps a hash footprint of files to prevent duplicate ingestion. Command files containing JSON are executed as DAG plans via `TaskExecutionContext`.
- **Edge cases**: Unreadable files, extremely large files causing OOM, permission errors moving files.
- **Dependencies**: psutil, core DAG executor, OCR libraries.

## 4. Autonomy Governor (Permissions & Risk Management)
- **Purpose**: Dynamically enforces safety and permission levels for tool execution to prevent dangerous autonomous operations.
- **Inputs**: Tool name, desired execution level (Levels 0 through 4).
- **Outputs**: Boolean approval or rejection, accompanied by a reasoning string.
- **User actions**: The user provides explicit consent for tools flagged as `WRITE_WITH_CONFIRM` (Level 3), or escalates the agent to `AUTONOMOUS` (Level 4).
- **Business rules**: 
  - Level 0: `CHAT_ONLY`
  - Level 1: `SUGGEST_ONLY`
  - Level 2: `READ_ONLY`
  - Level 3: `WRITE_WITH_CONFIRM`
  - Level 4: `AUTONOMOUS`
  Tools are dynamically categorized using a known registry or heuristic keyword matching (e.g., `write`, `delete`, `click`, `turn_on` imply write actions).
- **Edge cases**: Unrecognized tool names default to blocked unless dynamic keyword matching explicitly clears them for the current level.

## 5. Mission Scheduler
- **Purpose**: Manages delayed execution and automated retries of missions associated with goals.
- **Inputs**: Mission ID, Goal ID, delay seconds, maximum attempts, backoff factor.
- **Outputs**: A queue of `ScheduledMission` entries. Exposes `due()` entries to the main dispatcher loop.
- **User actions**: Implicitly scheduled by the agent or explicitly by user workflows.
- **Business rules**: Uses a pull-based model (no hidden asyncio tasks or background threads) where the main loop queries the scheduler. Implements exponential backoff for retries. Schedule state is serializable and persists across system restarts.
- **Edge cases**: Main loop stops pulling, maximum retry attempts exhausted.

## 6. Goal Manager
- **Purpose**: Owns the lifecycle of long-lived, high-level objectives (Goals) that span multiple missions and sessions.
- **Inputs**: Goal description, priority (1-10), parent goal ID, deadline, metadata.
- **Outputs**: Active goal selection (next highest priority goal), goal status tracking (`PENDING`, `ACTIVE`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED`).
- **User actions**: Users can view and manage goals via the `/goals` dashboard.
- **Business rules**: Sorts pending goals by priority (1 is highest) and creation timestamp. Goals can have parent-child hierarchical relationships. State persists across restarts.
- **Edge cases**: Conflicting goals, goals abandoned without corresponding mission updates.

## 7. Memory Subsystem
- **Purpose**: Persistently stores user preferences, conversational episodes, and RAG ingested documents for retrieval.
- **Inputs**: Conversational turns (User/Jarvis), discrete events, text payloads extracted from files/screens.
- **Outputs**: Search results (keyword and vector-based) across all memories via API or `/memory` dashboard.
- **User actions**: Users can query the `/memory` UI or ask Jarvis to recall information.
- **Business rules**: Stored in a SQLite DB using `WAL` journal mode for concurrency. RAG queries extract metadata like `source` and `score`. Automatically limits log growth.
- **Edge cases**: Corrupted SQLite database, overly broad queries returning too many results.
- **Dependencies**: SQLite3.

## 8. Built-in Core Tools
- **Purpose**: Provides Jarvis with native OS, file system, hardware, and GUI capabilities.
- **Inputs**: Tool-specific JSON arguments.
- **Outputs**: Command results, file modifications, system state changes.
- **User actions**: Implicitly called by Jarvis during reasoning.
- **Features**:
  - **System & Files**: `get_time`, `get_system_stats`, `list_directory`, `read_file`, `write_file_safe`, `sort_files`, `fast_search` (optimized multi-threaded search).
  - **Hardware**: `send_hardware_command`, `read_sensor`, `list_hardware_devices`, `ping_device`.
  - **Screen & GUI**: `capture_screen`, `find_text_on_screen`, `describe_screen`, `click`, `type_text`, `hotkey`, `clipboard_set`.
  - **Web**: `web_search`, `web_scrape`.
  - **Media**: `convert_file_format` (dynamically converts files to PDF, WebP, MP4, etc.).
- **Business rules**: File paths are sandboxed to allowed directories (e.g., `workspace`, `data`). GUI tools can be globally disabled via the `allow_gui_automation` configuration flag.
- **Dependencies**: `psutil`, `pyautogui`, `pillow`, `ffmpeg`.

## 9. External Integrations Catalog
- **Purpose**: Allows Jarvis to interact with and control third-party services.
- **Inputs**: API tokens and OAuth credentials via environment variables or config.
- **Outputs**: API interactions, message sending, device control.
- **User actions**: Configuring integration credentials.
- **Features**:
  - **Home Assistant**: `get_entity_state`, `turn_on_entity`, `set_thermostat`, `call_service`.
  - **Google Calendar / Calendar**: Read/Write events.
  - **Gmail / Email**: Fetch and send emails.
  - **Github**: Manage repositories, issues, and pull requests.
  - **Notion**: Read and write knowledge base pages.
  - **Spotify**: Play music, control playback, manage playlists.
  - **Telegram / WhatsApp**: Read and send instant messages.
  - **Weather**: Retrieve forecasts.
- **Business rules**: Integrations conditionally load based on the presence of required configuration variables. Sensitive domains in Home Assistant (e.g., `lock`, `alarm_control_panel`) force explicit confirmation.
- **Edge cases**: API rate limits, network timeouts, invalid credentials.
- **Dependencies**: `aiohttp` and integration-specific SDKs.

## 10. Voice Interface Subsystem
- **Purpose**: Enables speech-to-text and text-to-speech interaction for hands-free usage.
- **Inputs**: Microphone audio stream.
- **Outputs**: Speaker audio stream, transcribed text commands.
- **User actions**: Starting Jarvis with the `--voice` flag, speaking the wake word.
- **Business rules**: Uses a continuous audio loop to listen for a wake word. Once triggered, records the user's command, transcribes it via the STT module, processes the command, and reads the response aloud via the TTS module.
- **Edge cases**: Background noise triggering false positives, audio device unavailability.

## 11. Security & Authentication Module
- **Purpose**: Secures the dashboard, APIs, and critical application functions from unauthorized access.
- **Inputs**: Passwords, API Tokens, Session Tokens.
- **Outputs**: Session cookies, CSRF tokens, authentication verdicts.
- **User actions**: Logging in, providing an API token in headers.
- **Business rules**: Admin user is bootstrapped from `JARVIS_ADMIN_USER` and `JARVIS_ADMIN_PASSWORD` environment variables. Sessions expire after 12 hours (`SESSION_TTL_S = 43200`). Passwords are hashed using `bcrypt` (if available) or `PBKDF2-HMAC-SHA256` with 260,000 rounds. API tokens are hashed before storage to prevent leakage.
- **Edge cases**: Missing secret keys fallback to a warning and a default key.
- **Dependencies**: `bcrypt` (optional), `hashlib`, `hmac`.

## 12. Audit Logging & Health Verification
- **Purpose**: Ensures system integrity and provides observability into the agent's behavior.
- **Inputs**: CLI flags (`--health-check`, `--strict-health`, `--verify`).
- **Outputs**: Health reports, verified audit logs.
- **User actions**: Running startup checks or manually verifying logs via CLI.
- **Business rules**: The system performs a lightweight preflight health check. The `--verify` flag checks the cryptographic integrity of the secure audit logs, ensuring no log entries have been tampered with. If the audit verification fails, the system exits with a specific error code (`AUDIT_FAILED`).
- **Edge cases**: Tampered log files, corrupted disk space.
