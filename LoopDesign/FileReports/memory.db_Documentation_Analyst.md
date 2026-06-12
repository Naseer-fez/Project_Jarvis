# File Report: memory.db
**Role**: Documentation Analyst
**Target**: `d:\AI\Jarvis\memory\memory.db`

## Overview
SQLite3 database file acting as the primary structured memory store. It retains preferences, historical dialogue, episodic events, and action logs.

## Schema & API Contract
The database consists of the following tables (some of which appear to be iterative duplicates or migrations):

1. **preferences**
   - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
   - `key`: TEXT UNIQUE NOT NULL
   - `value`: TEXT NOT NULL
   - `updated_at`: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

2. **episodic_memory**
   - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
   - `event`: TEXT NOT NULL
   - `timestamp`: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   - `category`: TEXT

3. **conversation_history**
   - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
   - `user_input`: TEXT NOT NULL
   - `assistant_response`: TEXT NOT NULL
   - `timestamp`: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   - `session_id`: TEXT

4. **episodes**
   - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
   - `content`: TEXT
   - `category`: TEXT
   - `created_at`: TEXT
   - `timestamp`: TEXT DEFAULT ''

5. **facts**
   - `key`: TEXT PRIMARY KEY
   - `value`: TEXT
   - `category`: TEXT
   - `updated_at`: TEXT

6. **conversations**
   - `id`: INTEGER PRIMARY KEY
   - `user_input`: TEXT
   - `assistant_response`: TEXT
   - `session_id`: TEXT
   - `timestamp`: TEXT

7. **actions**
   - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
   - `action`: TEXT NOT NULL
   - `result`: TEXT
   - `success`: INTEGER NOT NULL DEFAULT 1
   - `metadata`: TEXT
   - `timestamp`: TEXT NOT NULL

## Assumptions & Design Patterns
- **Iterative Schema Evolution**: The presence of both `conversation_history` and `conversations`, as well as `episodic_memory` and `episodes`, strongly implies that the schema was evolved over time. The active records reside in `preferences` (4 rows), `episodic_memory` (2 rows), and `conversation_history` (1 row), indicating they are the "current" or "legacy" active tables depending on which version of the agent is running. The newly defined but empty tables (`episodes`, `facts`, `conversations`, `actions`) might be from a recent migration.
- **Key-Value Stores**: Both `preferences` and `facts` serve as key-value stores. `preferences` tracks simple properties (e.g., `favorite_drink: coffee`, `work_style: night owl`), while `facts` introduces a `category` dimension for broader knowledge graphing.
- **Action Tracking**: The `actions` table tracks agent operations, their success/failure states, and execution metadata, implying a feedback loop where the agent can review its own past operational success.
- **Data Types**: SQLite dynamic typing is used extensively; datetime fields are stored as `TEXT` or `TIMESTAMP` with `DEFAULT CURRENT_TIMESTAMP`. Note the inconsistency in `episodes` having both `created_at TEXT` and `timestamp TEXT DEFAULT ''`.

## Developer Notes
- **Reconstruction Readiness**: When reconstructing the data access layer, one must account for the duplicate semantic tables. A decision must be made whether to consolidate `episodic_memory` into `episodes` and `conversation_history` into `conversations`, or maintain them for backward compatibility.
- **Dependencies**: Depends on SQLite3. A WAL file (`memory.db-wal`) and SHM file (`memory.db-shm`) are present, indicating that Write-Ahead Logging is enabled for concurrent reads/writes.
