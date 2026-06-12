# Documentation Analysis: `chroma.sqlite3`

## Target
`d:\AI\Jarvis\data\chroma\chroma.sqlite3`

## Overview
This is the central SQLite metadata database for the Chroma vector database used by JARVIS to store embeddings.

## Schemas
*Extract of key tables representing Chroma DB schema:*
- `tenants`: Multi-tenancy support.
- `databases`: Contains the databases linked to tenants.
- `collections`: Embedding collections (unique by name and database_id).
- `segments`: Embedding segments linked to collections.
- `embeddings`: Links segments to embedding IDs with `seq_id`.
- `embedding_metadata`: Key-value metadata store for embeddings.
- Full-text search (FTS5) virtual tables: `embedding_fulltext_search`, `embedding_fulltext_search_data`, etc., for tokenized trigram search.
- Queue system: `embeddings_queue` handling async embedding tasks.

## Assumptions & Contracts
- This is an auto-generated schema from the `chromadb` python package.
- It provides a local, persistent vector store mechanism.
- Includes trigram-based full-text indexing out of the box via SQLite FTS5 for metadata or string contents.
- Multi-tenant architecture is supported natively by Chroma, although likely only a single tenant is used locally.

## Developer Notes
- This directory `chroma/` and its `chroma.sqlite3` should not be manually modified. All interactions must go through the Chroma API.
- The presence of this file confirms the usage of ChromaDB for semantic search/memory embeddings within JARVIS.
