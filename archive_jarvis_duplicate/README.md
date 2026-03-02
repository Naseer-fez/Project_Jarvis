# JARVIS - Session 5: Voice & Automation Loop
> Just A Rather Very Intelligent System  
> **Architecture: Trusted Core | 100% Offline | Deterministic**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    TRUSTED CORE                         │
│                                                         │
│  [Wake Word]──►[STT]──►[Controller]──►[Planner]         │
│  Porcupine    Whisper   StateMachine   DeepSeek-R1:8b   │
│                              │                          │
│                        [Risk Check]                     │
│                        Table-based                      │
│                              │                          │
│                        [Executor]                       │
│                              │                          │
│                         [TTS Reply]                     │
│                          Piper TTS                      │
│                                                         │
│  ┌────────────────────────────────────────┐             │
│  │         HYBRID MEMORY                 │             │
│  │  SQLite (facts) + ChromaDB (meaning)  │             │
│  │  Embeddings: all-MiniLM-L6-v2         │             │
│  └────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

## State Machine (Cannot be bypassed)

```
IDLE ──► LISTENING ──► TRANSCRIBING ──► PLANNING
                                            │
                                       RISK_CHECK
                                       /         \
                                  EXECUTING    RESPONDING
                                       \         /
                                        ──► IDLE
```

---

## Quick Start

### 1. Run Setup Script
```batch
setup_jarvis.bat
```

### 2. Get Free Porcupine Key
- Visit: https://console.picovoice.ai/
- Sign up free, copy your AccessKey
- Add to `.env`: `PORCUPINE_ACCESS_KEY=your_key`

### 3. Start Ollama
```batch
ollama serve
```

### 4. Launch Jarvis
```batch
cd D:\AI\Jarvis
.\jarvis_env\Scripts\activate
python main_v2.py
```

---

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Wake Word | pvporcupine | Detect "Jarvis" locally |
| STT | openai-whisper | Transcribe speech locally |
| Planner | DeepSeek-R1:8b | Generate JSON task plans |
| Risk Check | Table-scoring | Block dangerous operations |
| Memory | SQLite + ChromaDB | Facts + semantic recall |
| TTS | Piper / pyttsx3 | Speak responses locally |

---

## DO / DON'T

| ✅ DO | ❌ DON'T |
|-------|---------|
| Keep everything local | Use cloud APIs (OpenAI, Google, etc.) |
| Use Hybrid Memory for recall | Skip the Risk Evaluator |
| Use DeepSeek-R1:8b for planning | Bypass the State Machine |
| Use all-MiniLM-L6-v2 for embeddings | Jump from IDLE to EXECUTING |

---

## File Structure

```
D:\AI\Jarvis\
├── main_v2.py              # Entry point
├── requirements.txt
├── .env.example            # Copy to .env and fill keys
├── jarvis_memory.db        # SQLite database (auto-created)
├── chroma_db/              # ChromaDB vector store (auto-created)
├── logs/
│   └── jarvis.log
├── models/
│   └── piper/              # Download Piper voice model here
│       └── en_US-lessac-medium.onnx
├── core/
│   ├── state_machine.py    # Deterministic state transitions
│   ├── controller_v2.py    # Main orchestrator
│   └── risk_evaluator.py   # Table-based risk scoring
├── memory/
│   └── hybrid_memory.py    # SQLite + ChromaDB
├── voice/
│   └── voice_layer.py      # Porcupine + Whisper + Piper
└── tasks/
    └── task_planner.py     # DeepSeek-R1 JSON planner
```

---

## Downloading Piper Voice Model

```batch
cd D:\AI\Jarvis\models\piper
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en_US-lessac-medium.tar.gz
tar -xzf voice-en_US-lessac-medium.tar.gz
```

---

## Session History

| Session | Feature |
|---------|---------|
| 1-3 | Core state machine, basic planning |
| 4 | Semantic Memory (SQLite + ChromaDB) |
| **5** | **Voice Layer (Whisper + Piper + Porcupine)** |
| 6 (next) | Vision (llava), GUI dashboard |
