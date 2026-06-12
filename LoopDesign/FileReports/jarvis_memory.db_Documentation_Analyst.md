# Documentation Analysis: `jarvis_memory.db`

## Target
`d:\AI\Jarvis\data\jarvis_memory.db`

## Overview
This SQLite database stores the cognitive and episodic memory of the JARVIS agent, capturing facts, preferences, interactions, and actions over time.

## Schemas
1. **`facts` Table**
   - `key` (TEXT PRIMARY KEY)
   - `value` (TEXT NOT NULL)
   - `source` (TEXT NOT NULL DEFAULT 'user')
   - `created_at` (REAL NOT NULL)
   - `updated_at` (REAL NOT NULL)
   - `metadata` (TEXT NOT NULL DEFAULT '{}')
   - *Index*: `idx_updated` on `updated_at`
2. **`preferences` Table**
   - `key` (TEXT PRIMARY KEY)
   - `value` (TEXT)
   - `updated_at` (TEXT)
   - *Index*: `idx_preferences_updated_at` (DESC)
3. **`episodes` Table**
   - `id` (INTEGER PRIMARY KEY)
   - `event` (TEXT)
   - `category` (TEXT)
   - `timestamp` (TEXT)
   - *Index*: `idx_episodes_timestamp` (DESC)
4. **`conversations` Table**
   - `id` (INTEGER PRIMARY KEY)
   - `user_input` (TEXT)
   - `assistant_response` (TEXT)
   - `session_id` (TEXT)
   - `timestamp` (TEXT)
   - *Index*: `idx_conversations_timestamp` (DESC)
5. **`actions` Table**
   - `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
   - `action` (TEXT NOT NULL)
   - `result` (TEXT)
   - `success` (INTEGER NOT NULL DEFAULT 1)
   - `metadata` (TEXT)
   - `timestamp` (TEXT NOT NULL)
   - *Index*: `idx_actions_timestamp` (DESC)

## Assumptions & API Contracts
- **Timestamps**: Interestingly, `facts` uses `REAL` for timestamps (likely Unix epochs), whereas `preferences`, `episodes`, `conversations`, and `actions` use `TEXT` (likely ISO-8601 strings). This implies two different subsystems interacting with the database.
- **Metadata**: Both `facts` and `actions` tables contain a `metadata` column for extensible JSON configuration. `facts.metadata` defaults to `'{}'`.
- **Facts Source**: `facts` defaults to source `'user'`, implying the system might distinguish between user-provided facts and self-inferred facts.
- **Actions Logging**: The `actions` table explicitly tracks success/failure (`success` INTEGER DEFAULT 1) of agentic actions.

## Developer Notes
- This is the central repository for the "Hybrid Memory" system referenced in the startup logs (`hybrid_memory.py`). It marries semantic facts with episodic event streams.
