# Infrastructure Model

## Hardware Requirements
- **Compute**: Capable of executing native Python scripts. Relies heavily on host CPU and GPU (for `Ollama` models, `faster_whisper` transcriptions, and `chromadb` local embeddings).
- **Peripherals**: 
  - Microphone (required for `pvporcupine` wake-words).
  - Webcams / Desktop Display (required for `DesktopObserver` screenshot analysis).

## Third Party Cloud Infrastructure
- **LLM APIs**: Outbound TCP dependencies on OpenAI, Anthropic, or Groq (managed via `core.llm.cloud_client`).
- **External Webhooks**: Outbound dependencies to Github, Gmail, Notion, Spotify APIs defined in `integrations.clients`.