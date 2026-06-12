# File Report: jarvis.log
Role: Runtime Investigator

## Overview
This file is the main application log file for the Jarvis runtime. It captures initialization sequences, subsystem bootstrapping, HTTP requests, and runtime warnings/errors.

## Schemas & Contracts
The log entries follow a standard format:
`<Date> <Time>,<Millis> [<LEVEL>] <module.name>: <Message>`
Example: `2026-02-18 13:10:53,935 [INFO] core.controller: Initializing Jarvis (session: e49136bb)`

## Assumptions & API Contracts
- **Controller Initialization**: `core.controller` kicks off the startup routine and assigns a unique session ID (`e49136bb`).
- **Semantic Memory Subsystem**: `memory.semantic_memory` handles vector database connections and embeddings. It relies on the `all-MiniLM-L6-v2` model from the HuggingFace Hub.
- **Hybrid Memory Subsystem**: `memory.hybrid_memory` is configured to run in `HYBRID` mode, utilizing SQLite (for structured/relational storage) and ChromaDB (for vector search).
- **HuggingFace API Contract**: The system fetches model configurations, tokenizers, and weights dynamically from HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`).
- **Database Path**: ChromaDB is initialized at `D:/AI/Jarvis/data/chroma`.

## Execution Paths & State Transitions
- **Boot Sequence (Main Loop Prefix)**:
  1. `core.controller` initialized -> Application Start.
  2. Subsystem `memory.semantic_memory` initialized.
  3. Embedding Model requested -> HTTP GETs to `huggingface.co`.
  4. HuggingFace configuration and tokenizer resolved and cached.
  5. Vector database (`ChromaDB`) connection established.
  6. Memory Subsystem transitions to `HYBRID` mode state -> Boot sequence complete.

## Dependencies & Variables
- Dependencies: `httpx`, `huggingface_hub`, `sentence-transformers`, `chromadb`, `sqlite3`.
- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- ChromaDB Path: `D:/AI/Jarvis/data/chroma`.
- Session IDs are dynamically generated at launch.

## Prompts Found
None.
