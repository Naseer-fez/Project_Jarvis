# Project Reconstruction Blueprint

This blueprint serves as the definitive roadmap for a fresh engineering team to reconstruct the Jarvis platform from scratch using modern standards while preserving exact behavioral parity.

---

## 1. Project & Folder Structure

To maintain modularity and separation of concerns, the reconstructed project should adopt the following directory tree:

```text
jarvis_reborn/
├── cmd/
│   └── jarvis/             # Main entry points (CLI, Daemon)
├── internal/
│   ├── api/                # FastAPI / HTTP / WebSocket routes
│   ├── autonomy/           # Risk Evaluator, Governor, Scheduler
│   ├── controller/         # IntentRouter, ComplexityScorer, Orchestrator
│   ├── core/               # StateMachine, EventBus
│   ├── database/           # SQLite repositories & ChromaDB clients
│   ├── execution/          # DAG Executor, DispatchPipeline
│   ├── llm/                # LLM Dispatchers, ModelRouter
│   └── security/           # AuthManager, Crypto wrappers
├── pkg/
│   └── plugins/            # Plugin registry & integration interfaces
├── web/
│   ├── static/             # JS, CSS, ParticleCanvas
│   └── templates/          # HTML Jinja views
├── scripts/                # Bootstrappers (install.ps1, docker-compose)
└── docker/                 # Dockerfiles
```

---

## 2. Implementation Phasing

### Phase 1: Core Foundation & Security
1. **Initialize Data Stores**: Implement `memory.db` and `auth.db` SQLite schemas (refer to *Data Model Spec*).
2. **Authentication Layer**: Implement PBKDF2 hashing, CSRF generation, and the `AuthManager`.
3. **State Machine**: Build the global event bus and strict state transition matrix (refer to *Business Logic Spec*).

### Phase 2: Heuristics & Routing
1. **Complexity Scorer**: Implement the regex/keyword heuristic engine to score inputs from 0.0 to 1.0.
2. **Intent Router**: Build the fast-paths for explicit commands (web search, preferences, goal additions) to bypass LLMs.
3. **Semantic Memory**: Spin up the ChromaDB client and build the `ContextCompressor` logic.

### Phase 3: Agentic Execution (The Brain)
1. **LLM Orchestration**: Implement the model fallback logic and prompt templating.
2. **DAG Planner**: Build the parser that converts LLM JSON outputs into a Directed Acyclic Graph.
3. **Autonomy Governor**: Implement the Risk Matrix (Levels 0-4) blocking dangerous executions (refer to *Security Spec*).
4. **Tool Registry**: Scaffold the base interface for system tools.

### Phase 4: Desktop & External Interfaces
1. **Desktop Observer**: Implement PyAutoGUI / Vision modules to capture screen context and execute UI clicks.
2. **FastAPI Dashboard**: Reconstruct the Web UI, Jinja templates, and the 2-second heartbeat WebSocket.
3. **Live Automation**: Implement directory watchers for drop-in commands and RAG ingestion.

---

## 5. Technology Stack Recommendations

While the original is written in Python, the logic is highly transferrable.

- **Backend**: Python 3.12+ (FastAPI) or Go (Fiber/Gin) for better concurrency if performance becomes a bottleneck.
- **Frontend**: Vanilla JS with WebSockets is sufficient. If scaling UI complexity, consider HTMX or a lightweight framework like Svelte to replace Jinja.
- **Database**: SQLite (WAL mode) remains optimal for a local-first OS. ChromaDB is ideal for vectors, or alternatively, PostgreSQL with `pgvector` if moving to a cloud-hosted variant.
- **LLM**: Continue utilizing Ollama for local execution to ensure privacy, with OpenAI/Anthropic APIs purely as fallbacks.

## 6. Testing Requirements

1. **Unit Tests**: Coverage must be >85% for `complexity_scorer.py`, `risk_evaluator.py`, and `auth.py`.
2. **Integration Tests**: Mock LLM responses to test the DAG Executor pipeline and loop termination conditions.
3. **Security Audits**: Fuzz the WebSocket endpoints and API token headers. Validate directory traversal guards on `/api/view-file` and `/gui-audit/`.
