# 13_State_Management.md

## WHY does this subsystem exist?

The State Management subsystem exists to solve the fundamental architectural challenge of large language model (LLM) integration: LLMs are inherently stateless, amnesiac functions. In an asynchronous, multi-modal operating system like Jarvis, executing autonomous multi-step loops while simultaneously handling background user inputs and daemon triggers requires rigorous structural scaffolding. 

This subsystem exists to externalize the cognitive state of the agent into a physical, thread-safe, and crash-resilient format. Without it, the system would collapse under the weight of its own asynchronous complexity—background processes and direct user commands would collide, execution progress would vanish upon a restart, and the agent would endlessly fall into redundant execution loops due to a lack of situational awareness.

## WHAT responsibility does it own?

The State Management subsystem is the definitive source of truth for the agent's current lifecycle phase and memory persistence. Its core responsibilities are:

- **Topological State Constraints:** It rigidly governs execution flow using a predefined mathematical topology of states (e.g., `IDLE` -> `THINKING` -> `PLANNING` -> `RISK_EVALUATION` -> `EXECUTING`). It rejects invalid transitions via an `IllegalTransitionError` and maintains a forensic rolling 100-event audit trail of state changes.
- **Persistence & Hydration:** It owns the read-modify-write cycle for critical execution tracking files, ensuring state survives system reboots. This includes:
  - `goals.json`: Tracks long-term execution plans, scheduled tasks, and autonomous goals.
    ```json
    {
      "saved_at": "2026-06-11T12:31:51.413437+00:00",
      "goals": [],
      "schedule": []
    }
    ```
  - `automation_state.json`: Maintains a deduplication filter using 64-character SHA-256 `seen_fingerprints` to guarantee system idempotency.
    ```json
    {
      "saved_at": "2026-06-11T12:31:51.413437+00:00",
      "seen_fingerprints": [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      ],
      "stats": {
        "started_at": "2026-06-11T12:54:12.185831+00:00",
        "last_scan_at": "2026-06-11T12:56:48.020677+00:00",
        "last_error": "cannot access local variable 'stored' where it is not associated with a value",
        "scanned_files": 2367440,
        "ingested_files": 638,
        "ingested_chunks": 937,
        "commands_executed": 0,
        "failed_files": 0,
        "skipped_files": 2366802,
        "live_screen_updates": 704
      }
    }
    ```
- **Concurrency & Context Confinement:** It isolates execution contexts using context managers (`StateGuard`). This structure guarantees that if a process crashes or is cancelled, the state gracefully falls back to `ERROR` or `ABORTED` rather than leaving an orphaned lock.
- **Cross-Thread Safety:** It manages the delicate synchronization boundaries between the core Python asynchronous event loops and OS-level threads.

## HOW does it interact with the rest of the system?

The subsystem acts as a globally reactive backbone, heavily decoupled via Dependency Injection and the Event Bus:

- **Event Driven Decoupling:** The `StateMachine` does not directly instruct the UI or background tasks. Instead, every valid transition publishes a `state_transition` event to the `EventBus`. Independent modules (CLI spinners, UI dashboards, logging metrics) subscribe to these events, ensuring they react in perfect sync without blocking the main event loop.
- **Execution Orchestration:** When the framework initiates an external tool or an LLM pipeline, the executor wraps the call in a `StateGuard` context manager. This shifts the state to `ACTING` or `THINKING` and bounds the operation. 
- **Idempotent Ingestion:** During live automation or data ingestion, background workers query `automation_state.json`. If an incoming artifact's SHA-256 fingerprint matches an entry, the interaction is skipped. If new, the state is atomically updated.

## WHAT would break if removed?

Removing this subsystem would result in an immediate architectural collapse across multiple vectors:

- **Concurrency Collapse & Split-Brain States:** Background daemon tasks (like `GoalRunner`) and user-triggered direct loops would concurrently attempt to mutate SQLite databases or `goals.json`. This race condition would result in split-brain memory, persistent data corruption, and catastrophic deadlocks.
- **Amnesiac Redundant Loops:** Without the hash-based tracking inside `automation_state.json`, background automations would endlessly re-ingest the same screenshots, logs, and context window payloads, wasting API tokens and completely destroying logical continuity.
- **Orphaned Execution & OOM DOS:** Without `StateGuard` managing `asyncio.CancelledError` mapping to `ABORTED`, tasks failing mid-execution would orphan their state. Bypassing the 100-event rolling limit would trigger runaway memory ballooning, resulting in Out-Of-Memory (OOM) crashes.
- **Safety Boundary Failures:** The framework would lose the ability to shift into `AWAITING_CONFIRMATION` or `RISK_EVALUATION`. Consequently, high-risk code execution or adversarial payloads could bypass human oversight, executing destructive tasks blindly.

## HOW would it be rebuilt from scratch without source code?

If rebuilding the State Management subsystem, strict adherence to isolation and atomicity is required:

1. **Map the Topological State Matrix:** Define a thread-safe Enum representing all phases of execution (`IDLE`, `THINKING`, `PLANNING`, `RISK_EVALUATION`, `EXECUTING`, `AWAITING_CONFIRMATION`, `ERROR`, `ABORTED`). Construct an allowed-transitions matrix. Any attempt to skip directly from `IDLE` to `EXECUTING` without `PLANNING` or `RISK_EVALUATION` must throw a fatal exception.
2. **Implement Context-Managed Guards:** Build a `StateGuard` utility utilizing Python `contextvars` to bound every async task. Ensure that `__aenter__` shifts the state and `__aexit__` guarantees cleanup, reliably mapping exceptions like `asyncio.CancelledError` to an `ABORTED` state.
3. **Establish Atomic JSON Persistence:** Build serializers for `goals.json` and `automation_state.json`. The implementation *must* use `.tmp` atomic file swapping. Writing directly to JSON files invites corruption if the system crashes mid-write. Implement a `seen_fingerprints` SHA-256 hash-set for the automation state to track what data has already been processed.
4. **Engineer Dual-Lock Concurrency:** Address the sync-to-async boundary explicitly. Use a unified `asyncio.Lock` for event-loop bound mutations, but wrap shared state access paths in an OS-level `threading.RLock` to prevent deadlocks when reactive synchronous workers access the state machine simultaneously.
5. **Bind to the Event Bus:** Ensure that the state singleton is never mutated silently. The rebuilt state machine must accept an `EventBus` dependency and fire immutable payloads representing the `(old_state, new_state, trace_id)` on every successful state change.
