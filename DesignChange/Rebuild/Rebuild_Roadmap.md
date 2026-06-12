# Rebuild Roadmap

## Phase 1: Foundational Scaffolding
- Set up logging (`audit`).
- Define core interfaces (`ToolObservation`, `Capability`).
- Stand up the Dependency Injection Registry (`CapabilityRegistry`).

## Phase 2: Memory & State
- Setup `SQLitePool` and schema migrations.
- Setup `EmbeddingManager` and `chromadb` client.
- Build the `ContextCompressor` to manage context window limits.

## Phase 3: The Engine
- Implement `ModelRouter` and LLM SDK clients.
- Build the `AgentLoopEngine` (Observation -> Reflection -> Planning -> Action).
- Enforce the `AutonomyGovernor` rules.

## Phase 4: Integrations Expansion
- Re-implement all clients: Github, Gmail, Calendar, Spotify, Telegram, HomeAssistant.
- Map all external actions to `RiskLevel`s.

## Phase 5: Peripherals & Dashboard
- Implement FastAPI WebSockets.
- Implement Voice Layer (`pvporcupine`, STT, TTS).
- Implement Vision and Desktop Control (`pyautogui`).