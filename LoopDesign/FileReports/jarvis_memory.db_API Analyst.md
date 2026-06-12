# API Analyst Report: jarvis_memory.db

## Overview
This SQLite database manages the state, episodic memory, and interaction history of the Jarvis system.

## Schema / Structure
```sql
CREATE TABLE facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE preferences (
    key TEXT PRIMARY KEY, 
    value TEXT, 
    updated_at TEXT
);

CREATE TABLE episodes (
    id INTEGER PRIMARY KEY, 
    event TEXT, 
    category TEXT, 
    timestamp TEXT
);

CREATE TABLE conversations (
    id INTEGER PRIMARY KEY, 
    user_input TEXT, 
    assistant_response TEXT, 
    session_id TEXT, 
    timestamp TEXT
);

CREATE TABLE actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    result TEXT,
    success INTEGER NOT NULL DEFAULT 1,
    metadata TEXT,
    timestamp TEXT NOT NULL
);
```

## API Contracts & Data Schema
- **Knowledge Base APIs (`facts`)**: 
  - External endpoints storing facts will write to `key`/`value` schemas.
  - `metadata` defaults to an empty JSON string `'{}'`, indicating a JSON schema contract within a text field.
  - Timestamps (`created_at`, `updated_at`) are `REAL` (epoch floats).
- **Preference APIs (`preferences`)**: 
  - Key-Value store where `updated_at` is interestingly `TEXT`, differing from `facts`. API schemas returning timestamps must handle format variations (Epoch vs ISO 8601).
- **Episodic Memory APIs (`episodes`)**: 
  - Event schemas expect string data categorized by `category` with a `TEXT` timestamp.
- **Chat History APIs (`conversations`)**: 
  - APIs retrieving context will pull `user_input` and `assistant_response` paired by a `session_id`.
- **Action Tracing APIs (`actions`)**: 
  - Logs discrete system actions. `success` is boolean `INTEGER` (1 or 0). `metadata` is a stringified JSON schema.

## Assumptions
- Systems using `jarvis_memory.db` serialize complex structures into JSON before saving to `metadata`.
- The database does not use foreign keys connecting `conversations.session_id` to an explicit sessions table, meaning session identifiers are managed purely by the application layer.
