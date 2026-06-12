# Executive Summary

## System Identity & Purpose
The `Project_Jarvis` codebase represents a highly autonomous, modular AI assistant and "Local-First AI OS". It is designed to act as a unified proxy between a user (via voice, CLI, or web dashboard) and their local machine or third-party integrations. Unlike standard chatbots, Jarvis employs Directed Acyclic Graph (DAG) task planning, heuristic risk evaluation, and dynamic GUI automation (via PyAutoGUI/vision) to execute complex, multi-step workflows directly on the user's desktop.

## Core Capabilities
Jarvis's primary directive is to execute user intents accurately while safeguarding the local environment. It accomplishes this through:
- **Agentic Execution:** Translating ambiguous human goals into structured DAG plans.
- **Multi-modal IO:** Reacting to file-drops, desktop screenshots, active window observations, and direct text/voice commands.
- **Hybrid Cognition:** Fusing standard factual storage (SQLite) with semantic embedding search (ChromaDB) for high-context awareness.
- **Proactive Autonomy:** Utilizing background loops to schedule goals, evaluate state machines, and retry failed operations without prompting.

## Architecture Highlights
The system relies on a monolithic Python-based core orchestrator (`JarvisControllerV2`). The backend is served via FastAPI, streaming reactive state updates down to a vanilla HTML/JS frontend via WebSockets. It operates largely free of heavy external dependencies (no ORMs, no frontend frameworks like React, no external managed databases).

## Value Proposition for Reconstruction
This suite of specifications contains the exhaustive mapping of Jarvis's structural and behavioral DNA. By separating the logical bounds (Architecture, API, Database, Flows) from the domain logic (Business Rules, Risk Evaluators), a new engineering team can reliably reconstruct the intelligence platform without needing the source code, choosing to implement it in any modern language (e.g., Rust, Go, or a newer Python stack) while retaining 100% of its behavioral parity.
