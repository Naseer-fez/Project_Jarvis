# Jarvis — Local Agentic AI Assistant

A fully agentic, locally-run AI assistant powered by **Ollama** (Mistral, LLaMA 3, Phi-3, etc.).  
Implements the complete Session-8-Final spec: agent loop, planning, risk management, autonomy governor, memory, and full auditability.

---

## Architecture

```
main.py
└── MainController
    ├── StateMachine         (IDLE → LISTENING → THINKING → PLANNING → ...)
    ├── OllamaLLM            (multi-turn chat via Ollama /api/chat)
    ├── TaskPlanner          (LLM → structured JSON plan)
    ├── AgentLoopEngine      (analyze → plan → risk → [confirm] → execute → observe → reflect)
    │   ├── RiskEvaluator    (weighted-sum: intent + tool + profile + environment)
    │   └── AutonomyGovernor (LEVEL 0-3 permission enforcement)
    ├── ToolRouter           (sandboxed async tool dispatch + logging)
    ├── MemoryEngine         (episodic memory, JSONL persistence)
    └── AuditLogger          (append-only session artifacts)
```

---

## Setup

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (Mistral is the default)
ollama pull mistral
# or: ollama pull llama3, phi3, qwen2, etc.

# Start the server
ollama serve
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Jarvis

```bash
# Basic (text only, autonomy level 1 = suggest only)
python main.py

# With a different model
python main.py --model llama3

# Enable write tools (requires confirmation per write)
python main.py --autonomy 3

# Read-only tools, no confirmation needed
python main.py --autonomy 2

# Voice mode (requires optional voice deps)
python main.py --voice
```

---

## Autonomy Levels

| Level | Name              | What Jarvis can do                         |
|-------|-------------------|--------------------------------------------|
| 0     | Chat only         | Conversation only, no tools                |
| 1     | Suggest only      | Plans actions but never executes them      |
| 2     | Read-only tools   | Reads files, checks time/stats, searches memory |
| 3     | Write + confirm   | Can write files/logs **after your approval** |

---

## In-Session Commands

| Command        | Effect                              |
|----------------|-------------------------------------|
| `help`         | Show command reference              |
| `status`       | Show current state & configuration  |
| `memory`       | Show recent memory entries          |
| `clear`        | Clear conversation history          |
| `autonomy <N>` | Change autonomy level (0-3)         |
| `stop` / `cancel` | Interrupt running task           |
| `exit` / `quit` | End session                        |

---

## Agent Loop Cycle

Every action-oriented request runs through:

```
analyze_input → formulate_goal → generate_plan → evaluate_risk
→ [request_confirmation_if_required] → execute_single_step
→ capture_observation → evaluate_outcome → update_memory
→ decide_continue_or_stop
```

Hard stop conditions:
- Goal completed
- Risk threshold exceeded (composite score ≥ 0.75)
- User interrupt (`stop` / Ctrl+C)
- Unrecoverable tool failure
- Iteration limit (5 steps max)

---

## Risk Scoring

```
composite = intent(0.30) + tool(0.40) + profile(0.20) + environment(0.10)

< 0.40  → SAFE    (execute automatically)
0.40-0.60 → REVIEW  (log warning, proceed)
0.60-0.75 → CONFIRM (ask user before executing)
≥ 0.75  → BLOCK   (refuse, explain)
```

---

## Available Tools

| Tool              | Category | Risk  |
|-------------------|----------|-------|
| `get_time`        | system   | 0.00  |
| `get_system_stats`| system   | 0.05  |
| `list_directory`  | file     | 0.05  |
| `read_file`       | file     | 0.10  |
| `search_memory`   | memory   | 0.05  |
| `write_file_safe` | file     | 0.60  |
| `log_event`       | memory   | 0.20  |

All file tools are sandboxed to `./workspace/` and `./outputs/`.

---

## Session Artifacts

Every session writes to `./outputs/Jarvis-Session/`:

```
agent_plan.jsonl          — generated plans
execution_trace.jsonl     — full per-goal traces
tool_observations.jsonl   — tool call results
risk_assessment_log.jsonl — per-step risk scores
voice_interaction_log.txt — conversation transcript
autonomy_decisions.jsonl  — tool permission decisions
final_reflection.jsonl    — LLM reflections
memory_snapshot.json      — memory state at shutdown
```

---

## Extending Jarvis

### Add a new tool

```python
# In tools/builtin_tools.py or a new file:
async def my_custom_tool(param: str) -> str:
    return f"Result: {param}"

# Register it:
router.register("my_custom_tool", my_custom_tool)

# Add its risk score:
TOOL_RISK_MAP["my_custom_tool"] = 0.15
```

### Add voice support

Install the optional dependencies in `requirements.txt` (commented out), then implement:
- `voice/wake_word.py` — openwakeword listener
- `voice/stt.py` — faster-whisper transcription  
- `voice/tts.py` — pyttsx3 or edge-tts playback
- Wire into `MainController._get_input()` when `voice_enabled=True`

---

## Project Structure

```
jarvis/
├── main.py                    # Entry point
├── requirements.txt
├── README.md
├── core/
│   ├── controller.py          # MainController — wires everything
│   ├── state_machine.py       # FSM with explicit transition rules
│   ├── agent_loop.py          # AgentLoopEngine — the OODA cycle
│   ├── task_planner.py        # LLM → structured plan (JSON)
│   ├── tool_router.py         # Sandboxed async tool dispatch
│   ├── risk_evaluator.py      # Heuristic weighted-sum risk scoring
│   ├── autonomy_governor.py   # Permission enforcement (LEVEL 0-3)
│   ├── ollama_llm.py          # Ollama chat wrapper (multi-turn)
│   └── audit_logger.py        # Append-only session artifact writer
├── tools/
│   └── builtin_tools.py       # get_time, read_file, write_file_safe, etc.
├── memory/
│   └── memory_engine.py       # Episodic memory + JSONL persistence
├── voice/                     # (stub — implement for voice mode)
├── workspace/                 # Sandboxed working directory
└── outputs/                   # Session artifacts & logs
```
