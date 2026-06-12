# File Report: memory.db
**Role:** Data Model Analyst
**Target:** `d:\AI\Jarvis\memory\memory.db`

## Schema Overview & DB Architecture
This SQLite database is the primary persistent store for Jarvis's long-term and short-term memory, configuration, and audit logs. The DB operates using WAL (Write-Ahead Logging) mode, evidenced by the presence of `memory.db-wal` and `memory.db-shm` files. 

### Database Schema (Active Tables)

1. **`preferences`** (Active Data Found)
   - *Schema:* `id` (INTEGER PK AUTOINCREMENT), `key` (TEXT UNIQUE NOT NULL), `value` (TEXT NOT NULL), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
   - *Indexes:* `idx_preferences_updated_at` on `updated_at DESC`
   - *Sample Data:* `('communication_style', 'direct and concise')`, `('favorite_drink', 'coffee')`.
   - *Purpose:* Stores user specific or system specific key-value configuration flags and soft state.

2. **`episodic_memory`** (Active Data Found)
   - *Schema:* `id` (INTEGER PK AUTOINCREMENT), `event` (TEXT NOT NULL), `timestamp` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `category` (TEXT)
   - *Sample Data:* `('Built Jarvis memory system', '2026-02-16...', 'project')`
   - *Purpose:* Stores distinct factual events and milestones (episodes) in the AI's lifecycle or user interaction history. Categorized logically (e.g., "project").

3. **`conversation_history`** (Active Data Found)
   - *Schema:* `id` (INTEGER PK AUTOINCREMENT), `user_input` (TEXT NOT NULL), `assistant_response` (TEXT NOT NULL), `timestamp` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `session_id` (TEXT)
   - *Purpose:* Audit log and retrieval store for conversation turns. Maps user prompts to agent responses within specific sessions (`session_id`).

### Legacy / Inactive / Schema-Only Tables (No Data Found)
The following tables are defined but currently empty. They represent either legacy schema iterations or newly implemented modules that haven't recorded data yet:
1. **`episodes`**: `id` (PK), `content` (TEXT), `category` (TEXT), `created_at` (TEXT), `timestamp` (TEXT). *(Likely a deprecated duplicate or V2 of `episodic_memory`)*. Index: `idx_episodes_timestamp`.
2. **`facts`**: `key` (TEXT PK), `value` (TEXT), `category` (TEXT), `updated_at` (TEXT). *(Likely a deprecated duplicate or V2 of `preferences`)*.
3. **`conversations`**: `id` (PK), `user_input` (TEXT), `assistant_response` (TEXT), `session_id` (TEXT), `timestamp` (TEXT). *(Likely a legacy duplicate of `conversation_history`)*. Index: `idx_conversations_timestamp`.
4. **`actions`**: `id` (PK), `action` (TEXT), `result` (TEXT), `success` (INTEGER DEFAULT 1), `metadata` (TEXT), `timestamp` (TEXT NOT NULL). Index: `idx_actions_timestamp`. *(Likely tracks tool calls/action executions).*

## API Contracts & Dependencies
- Assumes consumers querying memory use recent timestamp sorts (hence the presence of multiple `DESC` indexes on timestamp fields).
- Active tables use Python's or standard SQL's `CURRENT_TIMESTAMP`, which is UTC by default. Noticeable format variation exists in active data updates (`'2026-02-16 23:24:53.778540'` vs `'2026-02-18T16:33:23.875927'`), indicating missing strict data sanitation on timestamps before insertion by the ORM/Data Layer.
- `session_id` in `conversation_history` indicates dependency on a session-management layer.

## State Objects & DTOs
Based on the schemas, the data model implies the following DTOs used in the application layer:
```python
class PreferenceDTO:
    id: int
    key: str
    value: str
    updated_at: datetime

class EpisodicMemoryDTO:
    id: int
    event: str
    timestamp: datetime
    category: str

class ConversationTurnDTO:
    id: int
    user_input: str
    assistant_response: str
    timestamp: datetime
    session_id: str
```

## Prompts & Configuration Variables
- Configuration Variable entries reside within the `preferences` table. E.g., `communication_style = 'direct and concise'`. This explicitly modifies LLM behavior at runtime, acting as injected configuration. 
- No raw foundational system prompts are stored directly in this database.
