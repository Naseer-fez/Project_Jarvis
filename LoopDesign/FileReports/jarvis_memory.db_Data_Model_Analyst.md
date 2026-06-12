# Data Model Analyst Report: jarvis_memory.db

## File Analysis
- **Filename**: `jarvis_memory.db`
- **Path**: `d:\AI\Jarvis\data\jarvis_memory.db`
- **Format**: SQLite3 Database

## Schema and State Objects
This is a central memory and state repository for Jarvis, storing facts, episodes, conversations, actions, and preferences.

### Tables
1. **`facts`**
   ```sql
   CREATE TABLE facts (
       key         TEXT PRIMARY KEY,
       value       TEXT NOT NULL,
       source      TEXT NOT NULL DEFAULT 'user',
       created_at  REAL NOT NULL,
       updated_at  REAL NOT NULL,
       metadata    TEXT NOT NULL DEFAULT '{}'
   )
   ```
   - Stores generalized long-term facts or Key-Value knowledge. Includes source tracking and JSON metadata string.

2. **`preferences`**
   ```sql
   CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)
   ```
   - Simple KV store for configuration strings. Note: `updated_at` is TEXT here, diverging from `facts` where it is REAL.

3. **`episodes`**
   ```sql
   CREATE TABLE episodes (id INTEGER PRIMARY KEY, event TEXT, category TEXT, timestamp TEXT)
   ```
   - Temporal episodic memory. Event is likely serialized JSON or a textual representation of what happened. Timestamp is TEXT.

4. **`conversations`**
   ```sql
   CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_input TEXT, assistant_response TEXT, session_id TEXT, timestamp TEXT)
   ```
   - Dialog history tracking. Grouped by `session_id`.

5. **`actions`**
   ```sql
   CREATE TABLE actions (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       action TEXT NOT NULL,
       result TEXT,
       success INTEGER NOT NULL DEFAULT 1,
       metadata TEXT,
       timestamp TEXT NOT NULL
   )
   ```
   - Logs actionable operations executed by the agent, capturing whether it was successful and output metadata.

## Assumptions & Contracts
- The schema uses inconsistent types for timestamps (e.g., `REAL` in `facts`, `TEXT` in `episodes`, `conversations`, `actions`, `preferences`).
- Default `source` for facts is 'user'. Metadata is expected to be valid JSON.
- Note: It was observed from crash logs that there are schema evolution issues with the `episodes` table lacking the `timestamp` column in older databases, implying a strict requirement for `timestamp` that older instances might violate without a migration.

## Dependencies & Variables
- Acts as the primary transactional datastore for agentic capabilities (memory, RAG, history, context window generation).

## Extracted Prompts
None found directly in the schema, but records stored inside `actions` and `conversations` represent prompt I/O states.
