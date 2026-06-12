# Service Map

Although deployed as a monolithic structure on the local host, the system functions conceptually as three interconnected services.

## 1. Jarvis Execution Daemon (`core`)
The primary service running the background event loop and state machine.
- **Responsibilities**: Goal tracking, LLM orchestrations, context compression, memory embedding retrieval, tool invocation.
- **Key Threads**: `AgentLoopEngine`, `BackgroundMonitor`, `LiveAutomationEngine`.

## 2. API & Dashboard Service (`dashboard`)
A FastAPI web server running over `uvicorn`.
- **Responsibilities**: Exposing the UI for users, receiving direct web requests, pushing WebSocket events to the front-end, handling clicker states.
- **Key Threads**: The main ASGI `asyncio` event loop.

## 3. Peripheral Hardware & Voice Layer
Services connecting to local microphones, speakers, and microcontrollers.
- **Responsibilities**: Wake word detection (`WakeWordDetector`), STT (`SpeechToText`), TTS (`TextToSpeech`), and Serial Device Control (`SerialController`).
- **Key Threads**: Usually dedicated daemon threads blocking on microphone chunks or serial read operations.