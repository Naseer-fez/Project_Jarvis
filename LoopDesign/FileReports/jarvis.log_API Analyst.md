# File Report: jarvis.log
**Role:** API Analyst

## Logs & External Contracts
The log file outlines the startup sequence of the `Jarvis` core and details various HTTP requests and subsystem initializations.

- **External Dependency: HuggingFace API**:
  - The system dynamically fetches model configurations, tokenizers, and weights for `sentence-transformers/all-MiniLM-L6-v2` via HTTP GET/HEAD requests to `https://huggingface.co`.
  - Rate limiting notice seen: "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads." This denotes an optional configuration variable `HF_TOKEN`.

- **Internal Subsystem API Structures**:
  - `core.controller`: Manages session initialization (e.g., session `e49136bb`).
  - `memory.semantic_memory`: Handles loading embedding models and connects to a vector database (ChromaDB) stored at `D:/AI/Jarvis/data/chroma`.
  - `memory.hybrid_memory`: Runs in HYBRID mode, coordinating between SQLite (relational storage) and ChromaDB (vector storage).

## Configuration Variables Identified
- `HF_TOKEN` (optional but recommended for HuggingFace Hub authenticated requests).
- Database Paths: Expects ChromaDB at `D:/AI/Jarvis/data/chroma`.
- Embedding Model: Hardcoded or configured to use `all-MiniLM-L6-v2`.

## API Contracts
- Connects to ChromaDB natively or via its local API.
- Assumes local filesystem availability for caching HuggingFace models under `huggingface.co/api/resolve-cache/models/...` (or relies on the standard `huggingface_hub` cache behavior).
