# API Analyst Report: controller\llm_orchestrator.py

## Dependencies
- `import asyncio`
- `import logging`
- `from typing import Any`
- `from typing import Dict`

## Schemas & API Contracts (Classes)

### Class `LLMOrchestrator`
**Methods:**
- `def __init__(self, llm_dispatcher: Any) -> None`
- `async def startup(self) -> None`
- `async def shutdown(self) -> None`
- `async def dispatch(self, text: str, classification: Dict[str, Any], session_id: str, trace_id: str) -> str`

