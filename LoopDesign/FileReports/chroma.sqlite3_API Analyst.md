# API Analyst Report: chroma.sqlite3

## Overview
This SQLite database represents the persistent metadata store for the Chroma Vector Database backend, tracking embeddings, collections, and segments used by the Jarvis system.

## Schema / Structure Overview
*This covers the core schema extracted, detailing vector storage APIs.*
- **`migrations`**: Tracks schema versions.
- **`tenants`** & **`databases`**: Multi-tenancy structure.
- **`collections`**: Manages logical embedding groupings (supports `config_json_str` and `schema_str`).
- **`segments`**: Links sub-collections.
- **`embeddings`**: The core mapping of `segment_id`, `embedding_id`, and `seq_id`.
- **`embedding_metadata`** & **`embedding_metadata_array`**: Dynamically typed KV storage linking string, int, float, and bool values to embeddings.
- **`embedding_fulltext_search_*`**: Virtual tables using `fts5` for trigram-based full-text searching over metadata strings.
- **`embeddings_queue`**: An event-driven ingestion schema processing queues.

## API Contracts & Data Schema
- **Vector API Contract**: This file implies ChromaDB is the underlying engine for semantic search. APIs operating on embeddings must interact with Chroma's native client, which abstractions over these tables.
- **Queue System**: `embeddings_queue` tracks operations via an `operation` integer and `topic` string. Vector data is stored as a BLOB (`vector`), with an associated `encoding` type. This indicates an asynchronous ingestion API pipeline.
- **Multi-Tenant Schema**: The database explicitly structures `tenants` > `databases` > `collections`. Downstream API logic calling the vector database will supply tenant or database boundaries, even if Jarvis operates as a single tenant by default.
- **Full Text Search (FTS)**: Uses SQLite's FTS5 engine configured for trigrams. APIs searching metadata fields leverage this for fuzzy text matching natively.

## Assumptions
- External systems should not write to this database directly; they must interact via the ChromaDB API.
- All embedding vectors are externally encoded (e.g. by an external ML service) and stored as BLOBs here.
