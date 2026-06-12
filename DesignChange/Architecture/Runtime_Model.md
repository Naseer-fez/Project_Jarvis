# Runtime Model

The Python execution model is carefully designed to avoid Global Interpreter Lock (GIL) stalls during heavy IO or AI operations.

## Main Entrypoint
- `root/main.py` is the execution anchor. It instantiates the Event Bus, initializes `sqlite3` and `chromadb` connections, and spins up background daemons.

## Threading Architecture
- **Event Bus Thread**: Distributes inter-system messages.
- **FastAPI Uvicorn Loop**: Main thread executing async handlers for HTTP and WebSockets.
- **Agent Loop Thread**: A perpetual `while True` loop that drains the goal queue and interacts with the LLM APIs synchronously or via threaded offloading.
- **Audio Listener Thread**: Runs `pvporcupine` in a tight loop monitoring the microphone.

## Fault Tolerance
- **Error Boundaries**: Tool execution (`core.registry.CapabilityRegistry.execute`) catches arbitrary exceptions from external integration plugins and converts them into `ToolObservation` errors. This prevents a bad third-party API from crashing the agent loop.
- **State Machine Protection**: `StateGuard` context managers prevent concurrent modification of the agent's internal state (e.g., trying to execute a task while already thinking).