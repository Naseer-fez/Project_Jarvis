# JARVIS - Fully Offline Personal AI Assistant

**Version:** 0.1 (Text Mode)  
**Status:** Core logic complete, voice integration pending

---

## WHAT IS JARVIS?

A fully offline personal AI assistant that:
- ✓ Runs entirely on your local machine
- ✓ Remembers you across sessions (persistent memory)
- ✓ Adapts responses based on learned preferences
- ✓ Has **zero** cloud or paid API dependency
- ✓ Works without internet after initial setup

---

## CURRENT CAPABILITIES (SESSION 3)

### ✓ IMPLEMENTED
- Text-based interaction loop
- Intent classification (command, question, memory store/recall)
- Persistent long-term memory (SQLite)
- Session-scoped short-term memory
- Conversation history tracking
- LLM integration via Ollama
- User preference learning

### ⏳ DEFERRED TO FUTURE SESSIONS
- Speech-to-Text (Whisper integration)
- Text-to-Speech (Piper integration)
- Voice interaction loop
- Vector-based semantic search (ChromaDB)
- Command execution (file/app launching)

---

## SYSTEM REQUIREMENTS

- **OS:** Windows 10/11 (with WSL for dev) or Linux
- **Python:** 3.10+
- **Storage:** ~10GB free on D: drive (for models/data)
- **RAM:** 8GB minimum, 16GB recommended
- **CPU:** Modern multi-core (GPU optional)

---

## INSTALLATION GUIDE

### 1. INSTALL SYSTEM DEPENDENCIES

#### Ollama (LLM Runtime)
```powershell
# Download from https://ollama.com
# Install and verify
ollama --version

# Pull the model (5.2 GB)
ollama pull deepseek-r1:8b
```

#### FFmpeg (for Whisper, future)
```powershell
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract to D:\AI\ffmpeg\
# Add to PATH: D:\AI\ffmpeg\bin
ffmpeg -version
```

### 2. SET UP PROJECT DIRECTORY

```powershell
# Create project structure on D: drive
mkdir D:\AI\Jarvis
cd D:\AI\Jarvis

# Copy all files from this build into D:\AI\Jarvis\
```

### 3. CREATE PYTHON VIRTUAL ENVIRONMENT

```powershell
# Create venv
python -m venv jarvis_env

# Activate (PowerShell)
.\jarvis_env\Scripts\Activate.ps1

# Install dependencies
pip install requests rich pydantic python-dotenv numpy chromadb
```

### 4. VERIFY WHISPER CACHE LOCATION

```powershell
# Ensure Whisper cache is redirected to D:
# Check if symlink exists at:
C:\Users\<YourUsername>\.cache\whisper

# If not, create symlink (run as Admin):
New-Item -ItemType SymbolicLink `
  -Path "C:\Users\<YourUsername>\.cache\whisper" `
  -Target "D:\AI\Jarvis\data\whisper"
```

---

## PROJECT STRUCTURE

```
D:\AI\Jarvis\
├── jarvis_env\           # Python virtual environment
├── core\
│   ├── __init__.py
│   ├── llm.py           # Ollama interface
│   ├── intents.py       # Intent classification
│   └── controller.py    # Main orchestration
├── memory\
│   ├── __init__.py
│   ├── long_term.py     # SQLite persistence
│   ├── short_term.py    # Session memory
│   └── memory.db        # Database file (created on first run)
├── voice\               # (Future: STT/TTS)
│   └── models\
├── data\                # Heavy data (D: only)
│   ├── whisper\         # Whisper model cache
│   ├── embeddings\      # Vector DB
│   └── logs\
├── config\
├── tests\
└── main.py              # Entry point
```

---

## USAGE

### START JARVIS

```powershell
# Activate environment
.\jarvis_env\Scripts\Activate.ps1

# Ensure Ollama is running (in another terminal)
ollama serve

# Run Jarvis
python main.py
```

### EXAMPLE SESSION

```
╔════════════════════════════════════════════╗
║                                            ║
║           JARVIS v0.1 - OFFLINE            ║
║      Personal AI Assistant (Text Mode)     ║
║                                            ║
╚════════════════════════════════════════════╝

Session ID: a3f7b2c1
LLM Status: ✓ Connected

Ready. How can I help?

You: remember I like coffee
Jarvis: ✓ I'll remember that: favorite_thing = coffee

You: my name is John
Jarvis: ✓ I'll remember that: name = john

You: what do I like to drink?
Jarvis: Based on what I know: coffee

You: status
Jarvis: System Status:
- Session ID: a3f7b2c1
- Uptime: 45.2 seconds
- Exchanges this session: 4
- Stored preferences: 2
- LLM available: Yes

You: exit
Jarvis: Goodbye! Session saved.

✓ Session data saved to memory.db
```

---

## AVAILABLE COMMANDS

| Command | Description |
|---------|-------------|
| `remember [fact]` | Store information about you |
| `what do I like...` | Recall stored information |
| `help` | Show available commands |
| `status` | Show system status |
| `exit` | End session and save |

---

## MEMORY SYSTEM

### Long-Term Memory (Persistent)
- **Storage:** SQLite (`memory/memory.db`)
- **Tables:**
  - `preferences` - User preferences/facts
  - `episodic_memory` - Timestamped events
  - `conversation_history` - Past conversations
- **Survives:** Across sessions and reboots

### Short-Term Memory (Session)
- **Storage:** RAM
- **Contains:** Current conversation context (last 20 turns)
- **Lifetime:** Current session only

---

## INTENT CLASSIFICATION

Jarvis automatically detects what you mean:

| Intent | Example |
|--------|---------|
| **Memory Store** | "remember I like coffee" |
| **Memory Recall** | "what do I like to drink?" |
| **Question** | "what is the weather?" |
| **Command** | "exit", "help", "status" |

---

## TROUBLESHOOTING

### "LLM not available"
- Check Ollama is running: `ollama serve`
- Verify model is pulled: `ollama list`
- Check port 11434 is accessible

### "Memory not persisting"
- Check `memory/memory.db` exists
- Verify write permissions on directory
- Check for error messages in console

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.10+)

---

## DATA LOCATION POLICY

**CRITICAL RULE:** No heavy data on C: drive.

| Data Type | Location |
|-----------|----------|
| Project code | `D:\AI\Jarvis\` |
| Python venv | `D:\AI\Jarvis\jarvis_env\` |
| Memory DB | `D:\AI\Jarvis\memory\memory.db` |
| Whisper cache | `D:\AI\Jarvis\data\whisper\` |
| Vector DB | `D:\AI\Jarvis\data\embeddings\` |
| Logs | `D:\AI\Jarvis\data\logs\` |

---

## TESTING INDIVIDUAL COMPONENTS

```powershell
# Test long-term memory
python memory/long_term.py

# Test short-term memory
python memory/short_term.py

# Test LLM interface
python core/llm.py

# Test intent classifier
python core/intents.py

# Test controller
python core/controller.py
```

---

## NEXT STEPS (SESSION 4+)

1. **Voice Integration**
   - Implement STT (Whisper)
   - Implement TTS (Piper)
   - Add voice loop to main.py

2. **Semantic Search**
   - Integrate ChromaDB
   - Add vector-based memory recall
   - Improve contextual responses

3. **Command Execution**
   - File/app launching
   - System information queries

4. **Advanced Memory**
   - Multi-turn context window
   - Episodic memory browsing
   - Memory visualization

---

## BUILD LOG

### Session 1 (Planning)
- Defined architecture
- Established D: drive policy
- Verified dependencies

### Session 2 (Setup)
- Installed Ollama + deepseek-r1:8b
- Redirected Whisper cache
- Set up Python environment

### Session 3 (Core Logic) ✓ CURRENT
- ✓ Built memory system (long-term + short-term)
- ✓ Implemented LLM interface
- ✓ Created intent classification
- ✓ Built main controller
- ✓ Text-only interaction loop working
- ✓ All tests passing

---

## CORE PRINCIPLES

1. **Stability over features** - Logic must be rock-solid before adding voice
2. **Memory over personality** - The system remembers, not just responds
3. **Offline over cloud** - Zero runtime dependency on internet
4. **Systems over demos** - Build architecture, not quick hacks

---

## LICENSE

Personal project. Use at your own discretion.

---

## CONTACT

This is a personal build log. No support infrastructure yet.

**Status:** Session 3 complete. Core logic operational. Voice deferred to Session 4.
