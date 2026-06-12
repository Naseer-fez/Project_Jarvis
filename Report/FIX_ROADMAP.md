# Remediation Roadmap

## Phase 1 - Critical blockers
Fix Phase 4 imports (Controller). Bridge IntegrationRegistry.

## Phase 2 - Build failures
Sync `jarvis.spec` with `requirements.lock`.

## Phase 3 - Runtime failures
Install missing voice dependencies (`faster-whisper`), Ollama connectivity. Fix Calendar race conditions.

## Phase 4 - Integration failures
Fix Google Calendar UTC override. Add path validation to computer control.

## Phase 5 - Architecture cleanup
Decouple `core` and `dashboard`. Unify model configurations. 

## Phase 6 - Optimization
Offload synchronous file I/O in async routes. 

## Phase 7 - Technical debt removal
Remove orphaned `audit` module. Upgrade deprecated `fpdf`.
