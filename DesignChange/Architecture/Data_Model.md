# Data Model

The data layer employs polyglot persistence to satisfy both transactional and semantic AI retrieval requirements.

## Relational Persistence (`sqlite3` / `aiosqlite`)
Stores highly structured telemetry and system configuration.
- **Profiles**: `core.profile.UserProfileEngine` stores communication preferences, constraints, and implicit memories extracted during runtime.
- **Goal State**: `core.autonomy.goal_manager` persists `Goal` structs preventing total amnesia across reboots.
- **Auth**: `core.security.auth.AuthManager` manages users, API keys, and salted bcrypt password hashes.

## Vector Search (`chromadb`)
Stores semantic information for Retrieval Augmented Generation (RAG).
- **Episodic Memory**: Raw conversation transcripts.
- **Fact Memory**: Hard facts extracted via the LLM into `core.memory.hybrid_memory`.
- **Code Indexer**: Abstract Syntax Tree representations and method signatures stored by `core.memory.code_indexer_service.CodeIndexerService` allowing the LLM to search its own codebase.

## Transient State (In-Memory)
- **`ContextCompressor`**: Aggregates tokenized inputs before feeding them to the LLM to prevent Context Window exhaustion.
- **Payload Extraction**: Binary data from tools (e.g. `core.automation.payload_extractor` parsing PDFs or images) is maintained temporarily on disk or in memory buffers.