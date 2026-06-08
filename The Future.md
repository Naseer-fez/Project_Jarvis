# The Future: Jarvis Improvement Roadmap

This document outlines the strategic phases for evolving and hardening the Jarvis architecture.

## Phase 1: Quick Wins (High Impact, Low Effort)
- **Controller Refactoring**: Decouple intent routes from `controller_v2.py` to keep the orchestrator lean. *(Completed)*
- **Planner Cleanup**: Remove hardcoded regex-based fallbacks from `TaskPlanner` to force the LLM to handle ambiguity natively. *(Completed)*
- **Memory Pruning**: Implement strict auto-pruning mechanisms for the SQLite memory pool to ensure long-term stability without degrading local performance.

## Phase 2: Medium Improvements (Architectural Cleanup)
- **Voice Layer Decoupling**: Isolate the Voice Layer (`core/voice`) so that TTS/STT dependencies do not clutter or bottleneck the core startup sequence.
- **RAG Ingestion Optimization**: Improve the `LiveAutomationEngine` to handle large repository indexing gracefully without causing massive I/O spikes.
- **Desktop Observer Refinement**: Enhance `core/desktop/actions.py` to be more resilient to screen resolution changes and unexpected UI popups.

## Phase 3: Major Upgrades
- **Execution Sandboxing**: Containerize the tool execution environment (e.g., using Docker) to create a safe sandbox, preventing the agent from making destructive changes directly to the host OS.
- **Multi-Agent Orchestration**: Transition the monolithic `AgentLoopEngine` into a distributed multi-agent system where distinct specialized workers (e.g., Researcher, Coder, Reviewer) communicate over an internal event bus.
- **Vision-Language Model (VLM) Integration**: Upgrade the Desktop Observer to utilize local VLMs (like `llava`) for true semantic visual understanding of the screen, rather than relying on coordinate-based clicks.

## Phase 4: Product & Developer Experience
- **Dashboard Enhancements**: Decouple the web dashboard from Flask/FastAPI server logic to support a standalone React/Next.js frontend.
- **Observability**: Introduce advanced token-usage analytics and latency telemetry directly into the dashboard.
- **Admin Tooling**: Build dedicated UIs for managing memory contexts, adjusting autonomy risk thresholds, and reviewing `RiskEvaluator` logs.

## Phase 5: AI & Security Hardening
- **Agent-on-Agent Evals**: Implement an automated LLM evaluator to score and validate execution plans before they are dispatched to the DAG engine.
- **Prompt Injection Defense**: Harden web search tools against malicious online payloads that could manipulate the agent's context window.
- **Context Compression**: Introduce advanced summarization techniques to pack more relevant historical data into the LLM context without exceeding token limits.
