# Data Model Analyst Report: chroma.sqlite3

## File Analysis
- **Filename**: `chroma.sqlite3`
- **Path**: `d:\AI\Jarvis\data\chroma\chroma.sqlite3`
- **Format**: SQLite3 Database

## Schema and State Objects
This file is the relational backend for ChromaDB vector database, storing collections, tenant metadata, segments, and embedding references (while actual vectors may reside in `.bin` files inside the UUID-named directories).

### Key Tables
- **`migrations`**: Tracks schema version (`dir`, `version`, `filename`, `sql`, `hash`).
- **`tenants`** & **`databases`**: Multi-tenancy structure `(id, name, tenant_id)`.
- **`collections`**: Metadata for vector collections `(id, name, dimension, database_id, config_json_str, schema_str)`.
- **`segments`** & **`segment_metadata`**: Logical groupings inside a collection `(id, type, scope, collection)`.
- **`embeddings`**: Core tracking for embeddings `(id, segment_id, embedding_id, seq_id, created_at)`.
- **`embedding_metadata`** & **`embedding_metadata_array`**: Key-value dynamic properties for embeddings supporting STR, INT, FLOAT, BOOL types.
- **`embedding_fulltext_search*`**: Virtual FTS5 tables for keyword indexing on metadata strings.
- **`embeddings_queue`**: Processing queue for vector indexing.

## Assumptions & Contracts
- Entirely managed by the ChromaDB library. Should not be manually modified.
- Serves as the structural map linking UUIDs and metadata to the flat `.bin` level-0 vector storage files in `d:\AI\Jarvis\data\chroma\<uuid>\`.
- Uses FTS5 for fast trigram tokenization string matching.

## Dependencies & Variables
- Tightly coupled with the `chromadb` pip package version currently active in the environment. Schema updates are bound to Chroma's migration scripts.

## Extracted Prompts
None found.
