# JARVIS V1 — TRUSTED CORE

> **Offline | Local | Deterministic | Human-in-the-loop**

---

## What This Is

Jarvis V1 is the **trusted core** — the spine that everything else will be built on.

- ✅ Async state machine (deterministic, no illegal transitions possible)
- ✅ Hybrid memory (SQLite facts + Chroma semantic recall)
- ✅ LLM planner (DeepSeek R1:8b → JSON plans, no execution)
- ✅ Risk evaluator (table-based, stateless, conservative)
- ✅ Vision ingestion (LLaVA — passive, read-only, no action triggers)
- ✅ Immutable audit logs
- ❌ No voice (V2)
- ❌ No desktop automation (V3)
- ❌ No robotics (V3)

---

## Prerequisites

1. **Python 3.11+**
2. **Ollama** running locally:
   ```bash
   ollama serve
   ollama pull deepseek-r1:8b
   ollama pull llava
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure

```
D:/AI/Jarvis/
│
├── main.py                          # Entry point
├── requirements.txt
├── config/
│   └── jarvis.ini                   # All configuration here
│
├── core/
│   ├── controller.py                # THE SPINE — async event loop
│   ├── state_machine.py             # Deterministic FSM
│   ├── risk_evaluator.py            # Table-based risk scoring
│   ├── logger.py                    # Immutable audit + app logs
│   │
│   ├── memory/
│   │   └── hybrid_memory.py         # SQLite + Chroma facade
│   │
│   ├── planning/
│   │   └── task_planner.py          # DeepSeek R1:8b → JSON plans
│   │
│   ├── tools/
│   │   └── vision.py                # LLaVA vision (passive only)
│   │
│   └── hardware/
│       └── serial_controller.py     # STUB — blocked until V3
│
├── tests/
│   └── test_v1_acceptance.py        # V1 pass/fail checklist
│
├── data/                            # SQLite + Chroma data (auto-created)
└── logs/                            # Audit + app logs (auto-created)
```

---

## Running Jarvis V1

```bash
cd D:/AI/Jarvis
python main.py
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `<any text>` | Generate a plan for your intent |
| `vision <path>` | Analyze an image with LLaVA |
| `status` | Show system status |
| `memory` | Show stored facts |
| `history` | Show session history |
| `reset` | Reset from ERROR/ABORTED |
| `help` | Show help |
| `quit` | Shutdown |

---

## Running Tests

```bash
cd D:/AI/Jarvis
pytest tests/test_v1_acceptance.py -v
```

---

## V1 Acceptance Checklist

V1 is **DONE** only if ALL of these pass:

| Test | Criteria |
|------|----------|
| State Machine | Cannot be broken — illegal transitions always raise |
| Planner | Outputs valid JSON every time |
| Risk Evaluator | Blocks forbidden actions (physical, shell, unknown) |
| Vision | Never triggers actions — passive only |
| Memory | Recalls correctly — never hallucinate |
| Logs | Complete and immutable |
| Unknown requests | System says "I don't know" rather than guessing |

**If even one ❌ → V1 is not done.**

---

## Authority Model (V1)

```
L0_OBSERVE   → vision.py, memory.read
L1_INTERPRET → task_planner.py, risk_evaluator.py, memory.write
L2_INTERACT  → BLOCKED (V2)
L3_ACT       → BLOCKED (V3)
```

---

## V2 / V3 Roadmap

| Version | Adds |
|---------|------|
| **V1** (now) | Core: State machine, memory, planner, vision |
| **V2** | Wake word, continuous STT/TTS voice loop |
| **V3** | Desktop automation, serial robotics, vision-action loop |

> **If V1 is solid → V3 is safe. If V1 is rushed → V3 is dangerous.**
