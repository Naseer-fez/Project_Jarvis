# Folder Analysis: core

## Folder Purpose
Contains components related to core.

## Findings
- **JARVIS-CORE-001** (High): The `current_classification` attribute is stored directly on the `JarvisControllerV2` singleton instance inside the async `process()` method without locking. Because `process()` yields execution to an intent router before the classification is consumed by `_dispatch_llm`, concurrent user requests will overwrite each other's classification state.
- **JARVIS-CORE-002** (Medium): Synchronous file I/O operations block the main asyncio event loop across multiple components. Specifically, updating profiles, persisting goal states, and saving background automation states write directly to disk without offloading the work to threads.
- **JARVIS-CORE-003** (Medium): The intent handler for "automation" commands expects the `live_automation` engine to be mounted directly on the context (`ctx.live_automation`). However, `controller_v2.py` wraps it in an `automation_manager` subsystem, leaving the handler trying to extract `None`.
- **JARVIS-CORE-004** (Low): The CLI execution loop does not wrap the invocation of `await self.process(text)` in a `try/except` block.

## Risks & Dependencies
See full project roadmap.
