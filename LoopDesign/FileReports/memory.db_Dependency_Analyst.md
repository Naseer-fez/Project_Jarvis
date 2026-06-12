# Dependency Analyst Report: memory.db (including -shm and -wal)

## 1. Overview
This is the primary local SQLite database used by the system for storing interaction history, episodic memory, facts, and system actions. The presence of `memory.db-shm` and `memory.db-wal` confirms the database is operating in Write-Ahead Logging (WAL) mode.

## 2. Dependencies & Libraries
- **Format**: SQLite 3 Database.
- **Library Requirements**: An SQLite driver (e.g., Python's `sqlite3`, Node's `sqlite3`, etc.) that supports WAL journaling.
- **Service Dependencies**: Completely local file-based database, no network database dependencies.

## 3. Schema & API Contract
The following DDL schema dictates the API contract for the data access layer:

1. `preferences` (id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT, updated_at TIMESTAMP)
2. `episodic_memory` (id INTEGER PRIMARY KEY, event TEXT, timestamp TIMESTAMP, category TEXT)
3. `conversation_history` (id INTEGER PRIMARY KEY, user_input TEXT, assistant_response TEXT, timestamp TIMESTAMP, session_id TEXT)
4. `episodes` (id INTEGER PRIMARY KEY, content TEXT, category TEXT, created_at TEXT, timestamp TEXT)
5. `facts` (key TEXT PRIMARY KEY, value TEXT, category TEXT, updated_at TEXT)
6. `conversations` (id INTEGER PRIMARY KEY, user_input TEXT, assistant_response TEXT, session_id TEXT, timestamp TEXT)
7. `actions` (id INTEGER PRIMARY KEY, action TEXT, result TEXT, success INTEGER, metadata TEXT, timestamp TEXT)

- **Implicit API Contract**: Services must query this database for short/long-term memory retrieval and user fact updates. The split between `conversations` and `conversation_history` implies a possible schema migration or duplicated storage that data layers must handle.

## 4. Hidden Execution Links
- Execution relies on SQLite WAL mechanisms for concurrency. 
- Trigger systems or background pruning processes might rely on `timestamp` / `updated_at` fields across these tables to manage memory size or context windows.

## 5. Configuration Variables & Prompts
- No prompts found directly in the schema (the actual content of the database could contain prompts if facts/actions were seeded with them, but none are statically defined in the file schema).
- The `preferences` table implies a dynamic configuration dictionary stored in the DB (e.g., `key`/`value` pairs).
