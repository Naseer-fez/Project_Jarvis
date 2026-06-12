# Dependency Analysis: chroma.sqlite3

## Overview
This file is the SQLite backend for the ChromaDB vector database. It stores collection definitions, embedding metadata, and tenant mappings.

## Schemas / API Contracts
Contains precise standard ChromaDB system tables including:
- `collections`, `collection_metadata`, `databases`, `tenants`, `segments`, `segment_metadata`
- `embeddings`, `embedding_metadata`, `embedding_fulltext_search` (using FTS5 trigram)
- `migrations`

## Assumptions & Dependencies
- Hard dependency on **ChromaDB**. The specific table structures (`embedding_fulltext_search` with trigrams) confirm ChromaDB usage.
- Depends on `sqlite3` driver with FTS5 module enabled.
- Assumes vector payloads (e.g., `.bin` files) are stored on disk parallel to this DB in the `chroma/` directory tree (uuids).
