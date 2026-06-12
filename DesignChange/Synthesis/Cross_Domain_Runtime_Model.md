# Cross-Domain Runtime Model

The Jarvis system operates in a hybrid runtime mixing `asyncio` loops with dedicated thread pools.

## Execution Loops

### 1. FastAPI ASGI Event Loop (`dashboard`)
- Hosted by `uvicorn` in `dashboard.server`. 
- Completely asynchronous. Handles `WebSocket` streaming and REST requests.
- **Boundary Restriction**: Does not perform heavy computations. It uses `asyncio.create_task` or queues to offload work to `core`.

### 2. Autonomous Agent Loop (`core.agent.agent_loop.AgentLoopEngine`)
- Driven by a background worker or threaded executor.
- Uses `core.controller.goal_runner.GoalRunner` to manage the DAG of subtasks.
- Frequently halts execution waiting for user confirmation (via `core.autonomy.autonomy_governor`), shifting the state and emitting a message back to the ASGI layer.

### 3. Integration Offloading (`integrations`)
- Most tools in `integrations` (e.g., `github`, `gmail`, `home_assistant`) are wrapped in `asyncio` or executed via ThreadPoolExecutors if they rely on blocking libraries (like `requests` or `imaplib`).

### 4. Perception and Desktop Observation (`core.desktop`)
- `DesktopObserver` and `DesktopActionExecutor` run synchronous OS-level commands (e.g., `pyautogui`, `pygetwindow`). These are isolated in separate threads to prevent blocking the `asyncio` event bus.

## Synchronization Mechanisms
- **`core.runtime.event_bus`**: The central nervous system for pub/sub events.
- **`core.state_machine`**: Uses strict locks to prevent the agent loop and the dashboard from colliding on state transitions.