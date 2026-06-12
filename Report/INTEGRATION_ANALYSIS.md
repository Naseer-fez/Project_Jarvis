# Integration Analysis

### LOGS-002 (High)
**Files:** d:\AI\Jarvis\logs\app.log, d:\AI\Jarvis\logs\audit.jsonl.bak
**Description:** The application repeatedly fails to connect to Ollama and cloud LLM providers, leading to timeout errors and planner degradation.

### 2 (Critical)
**Files:** core/runtime/bootstrap.py, core/registry/registry.py, integrations/registry.py, core/controller/services.py
**Description:** Integrations (e.g., GitHub, Notion, Calendar) are successfully loaded into an isolated `IntegrationRegistry` at startup. However, the core execution engine and task planner exclusively query `CapabilityRegistry` (`tool_router`). These two registries are never bridged, rendering the integrations invisible to the LLM.

