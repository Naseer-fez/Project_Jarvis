# Manifest: `data` Folder

**High-Level Purpose**:
This directory acts as the local persistence layer and runtime state storage for Project Jarvis. It encompasses a wide array of stateful items, including relational databases, vector database storage (Chroma), LLM orchestration traces and missions, local AI model weights (Whisper, ONNX voices), application logs, and structured user data (goals, calendar).

**File and Subfolder Manifest**:

| Item | Type | Required Tier 2 Specialists | Description / Purpose |
|------|------|-----------------------------|-----------------------|
| `agentic/` | Dir | Runtime Investigator, Prompt Recovery Specialist | Root directory for execution traces, agent missions, and task history. |
| `agentic/missions/` | Dir | Prompt Recovery Specialist, Runtime Investigator | Contains records of agentic plans, tasks, or assigned autonomous objectives. |
| `agentic/traces/` | Dir | Runtime Investigator, Data Model Analyst | Contains execution logs, sub-agent communication traces, and AI decision pathways. |
| `chroma/` | Dir | Data Model Analyst, API Analyst | Local storage for ChromaDB (vector database), used for semantic search, document retrieval, or RAG. |
| `chroma/chroma.sqlite3` | File | Data Model Analyst | Primary relational metadata store for the local Chroma vector database. |
| `chroma/[uuid-folders]/` | Dir | Data Model Analyst | Individual Chroma collection index directories. Contains binary search indices (`data_level0.bin`, `header.bin`, `length.bin`, `link_lists.bin`, `index_metadata.pickle`). |
| `embeddings/` | Dir | Data Model Analyst, Dependency Analyst | Used for caching or storing local serialized text embeddings. |
| `logs/` | Dir | Runtime Investigator | Storage for application, error, and debug logging outputs. |
| `logs/jarvis.log` | File | Runtime Investigator | Primary runtime log file capturing overall system execution events and errors. |
| `voices/` | Dir | Dependency Analyst, Data Model Analyst | Local storage for Text-To-Speech (TTS) models. |
| `voices/en_US-lessac-medium.onnx` | File | Dependency Analyst, API Analyst | Local ONNX model weights for voice generation. |
| `voices/en_US-lessac-medium.onnx.json` | File | Data Model Analyst, Configuration Analyst | Configuration metadata and parameters for the ONNX voice model. |
| `whisper/` | Dir | Dependency Analyst, API Analyst | Local storage for Automatic Speech Recognition (ASR) models. |
| `whisper/small.pt` | File | Dependency Analyst, API Analyst | PyTorch model weights for the locally hosted Whisper speech recognition model. |
| `auth.db` | File | Data Model Analyst, API Analyst, Configuration Analyst | SQLite database handling secure authentication tokens or credentials. |
| `calendar.ics` | File | Data Model Analyst, API Analyst | Standard iCalendar format file storing scheduled events or user timetable data. |
| `goals.json` | File | Prompt Recovery Specialist, Data Model Analyst | Structured user goals, directives, or persistent long-term memory for overall agent objectives. |
| `jarvis_memory.db` | File | Data Model Analyst, API Analyst, Runtime Investigator | Primary SQLite relational database for long-term agent memory or interaction history. |
| `jarvis_memory.db-shm` | File | Data Model Analyst | SQLite Write-Ahead Log (WAL) shared memory file. |
| `jarvis_memory.db-wal` | File | Data Model Analyst | SQLite Write-Ahead Log (WAL) data file. |
| `.gitkeep` | File | Configuration Analyst | Placeholder to ensure the empty aspects of the `data` directory structure are tracked by version control. |
