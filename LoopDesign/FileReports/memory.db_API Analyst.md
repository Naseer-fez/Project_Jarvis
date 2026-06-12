# API Analyst Report: memory.db

## 1. File Overview
- **File**: `d:\AI\Jarvis\memory\memory.db`
- **Purpose**: Primary SQLite database representing the persistent storage API for the memory subsystem.

## 2. API Contract & Data Schema
The database tables define the internal structure for persisting memory elements:

**Table: `preferences`**
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `key` (TEXT UNIQUE NOT NULL)
- `value` (TEXT NOT NULL)
- `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)

**Table: `episodic_memory`**
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `event` (TEXT NOT NULL)
- `timestamp` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- `category` (TEXT)

**Table: `conversation_history`**
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `user_input` (TEXT NOT NULL)
- `assistant_response` (TEXT NOT NULL)
- `timestamp` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- `session_id` (TEXT)

**Table: `episodes`**
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `content` (TEXT)
- `category` (TEXT)
- `created_at` (TEXT)
- `timestamp` (TEXT DEFAULT '')

**Table: `facts`**
- `key` (TEXT PRIMARY KEY)
- `value` (TEXT)
- `category` (TEXT)
- `updated_at` (TEXT)

**Table: `conversations`**
- `id` (INTEGER PRIMARY KEY)
- `user_input` (TEXT)
- `assistant_response` (TEXT)
- `session_id` (TEXT)
- `timestamp` (TEXT)

**Table: `actions`**
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `action` (TEXT NOT NULL)
- `result` (TEXT)
- `success` (INTEGER NOT NULL DEFAULT 1)
- `metadata` (TEXT)
- `timestamp` (TEXT NOT NULL)

## 3. Assumptions & Dependencies
- The DB acts as the main local storage API for the application.
- Some tables appear redundant or evolutionary (`conversation_history` vs `conversations`, `episodic_memory` vs `episodes`), implying legacy support or separate module schemas mapping to the same DB.
- Indexes: `idx_preferences_updated_at`, `idx_episodes_timestamp`, `idx_conversations_timestamp`, `idx_actions_timestamp` suggest queries frequently filter/sort by time.
- Uses standard SQLite constructs, implying synchronous/local access rather than distributed.

## 4. Prompts found
- None.
