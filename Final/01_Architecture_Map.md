# Architecture Map

## Overview
Project Jarvis is a modular, multi-modal AI agent platform built in Python. It features a dashboard, integrations, hardware support, voice control capabilities, and an autonomous state machine. The Jarvis system is built as an asynchronous Python application featuring multiple modular components including `core`, `config`, `integrations`, `plugins`, `workflows`, `dashboard`, and `memory`.

## Subsystems and Core Components

1. **Core Subsystem** (`core/`)
   - `runtime`: Application bootstrap, entrypoint, configuration resolution, environment paths, fault diagnostics, exception hooks, shutdown coordination, and the initial model inventory.
   - `controller`: The primary orchestrator handling runtime events and state (`controller_v2.py`). It governs high-level execution cycles.
   - `state_machine`: Autonomous decision-making and flow control (`state_machine.py`).
   - `introspection` & `logging`: Health checks, monitoring, diagnostics, and audit trails (verified at startup if requested).
   - `llm`, `memory`, `plugins`, `ops`, `planner`, `voice`, `security`, `desktop`: Domain-specific processing blocks.
2. **Dashboard** (`dashboard/`)
   - Graphical or Web GUI components, running an internal GUI server (e.g., via `DashboardRuntime`) binding to a local host/port.
3. **Data & Storage** (`data/`, `chroma_db/`)
   - Local vector storage, file-based dumps. Database components (`sqlite`, `chroma_db`).
4. **Configuration Layer** (`config/`)
   - Stores `jarvis.ini` and environment (`.env`, `settings.env`) settings.
5. **Plugins & Integrations** (`plugins/`, `integrations/`)
   - Dynamic external module loading. Loaded early in the runtime boot via `IntegrationLoader` and registered into the `integration_registry`.
6. **Workflows** (`workflows/`)
   - Higher-level orchestrated tasks and pipelines.

## System Boundaries
- **Entrypoints**:
  - CLI: `python main.py` or `Start.ps1`
  - Modes: Headless, Voice, GUI (Dashboard), Verify, Health Check.
- **Config**: `config/jarvis.ini`, `.env`, `settings.env`
- **Dashboard**: Runs an internal GUI server (`core.runtime.dashboard_runtime`) binding to `127.0.0.1:7070` by default.
- **Memory**: Database components (`sqlite`, `chroma_db`).

## Component Coupling Warning
- The entrypoint dynamically loads the controller class and dashboard based on config.
- Logging subsystem dynamically injected in bootstrap.
