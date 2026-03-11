# JARVIS V2 — DEPLOYMENT CHECKLIST

**Version:** 2.0.0  
**Runtime:** V2 (controller_v2 + HybridMemory + Dashboard)  
**Date:** March 2026

---

## FILE INTEGRITY — Current Runtime

Verify these files exist in `D:\AI\Jarvis\`:

```
D:\AI\Jarvis\
├── config\
│   └── jarvis.ini                 ← primary configuration
├── core\
│   ├── controller_v2.py           ← active controller
│   ├── memory\
│   │   ├── hybrid_memory.py       ← SQLite + Chroma memory
│   │   ├── semantic_memory.py
│   │   └── sqlite_pool.py
│   └── introspection\
│       └── health.py
├── dashboard\
│   └── server.py                  ← FastAPI dashboard
├── data\
│   ├── jarvis_memory.db           ← SQLite (created on first run)
│   └── goals.json                 ← goal persistence (created on first run)
├── .env                           ← secrets (gitignored)
├── .env.example                   ← template for secrets
├── main.py                        ← entry point
└── requirements.txt
```

---

## PRE-DEPLOYMENT CHECKS

- [ ] Python 3.11+ installed (`python --version`)
- [ ] D: drive has at least 10 GB free space
- [ ] Ollama installed and running (`ollama serve`)
- [ ] At least one model pulled (`ollama list`)

---

## INSTALLATION

### Phase 1: Ollama
- [ ] `ollama pull deepseek-r1:8b` completed (~5.2 GB)
- [ ] `ollama list` shows the model
- [ ] `ollama serve` running in a dedicated terminal

### Phase 2: Python Environment
```powershell
cd D:\AI\Jarvis
python -m venv jarvis_env
jarvis_env\Scripts\activate
pip install -r requirements.txt
```
- [ ] No error messages during installation
- [ ] `(jarvis_env)` visible in the prompt

### Phase 3: Configure Secrets
Copy `.env.example` to `.env` and fill in values:
```
JARVIS_DASHBOARD_TOKEN=<strong-random-secret>
JARVIS_ENV=production
```
> [!CAUTION]
> Never leave `JARVIS_DASHBOARD_TOKEN` as the default `jarvis`.
> The server will log a SECURITY warning at startup if it detects the default.

---

## FIRST RUN VERIFICATION

### Health Check (fast — does not load LLM)
```powershell
python main.py --health-check
```
Expected output:
```
Health Report — ✅ HEALTHY (3/3 OK, 0 warn, 0 fail)
  ✅ config_loaded: True
  ✅ ollama_reachable: True
  ✅ memory_sqlite: True
```
Exit code 0 = ready. Exit code 1 = a check failed (see log output).

### Standard Start
```powershell
python main.py
```
- [ ] No `CRITICAL` or `ERROR` lines in stderr
- [ ] `Session …` appears in the console
- [ ] Respond to `status` and `help` commands

### Dashboard Start
```powershell
python main.py --gui --dashboard-token <your-token>
```
- [ ] `Dashboard: http://127.0.0.1:7070` in output
- [ ] `curl -H "X-Dashboard-Token: <your-token>" http://127.0.0.1:7070/health` returns `{"ok": true, ...}`
- [ ] `curl http://127.0.0.1:7070/` (no token) returns HTTP 401

---

## STORAGE PATHS

| Artifact       | Path (relative to project root) | Config key                    |
|---------------|----------------------------------|-------------------------------|
| SQLite DB     | `data/jarvis_memory.db`          | `[memory] sqlite_file`        |
| Goals JSON    | `data/goals.json`                | `[memory] goals_file`         |
| Chroma DB     | `data/chroma/`                   | `[memory] chroma_dir`         |
| App log       | `logs/app.log`                   | `[logging] app_file`          |
| Audit log     | `logs/audit.jsonl`               | `[logging] audit_file`        |

All paths are created automatically on first run. Override any path in `jarvis.ini`.

---

## FUNCTIONAL TESTS

### Test 1: Memory Storage
```
You: remember I like coffee
Expected: "I will remember you like coffee."
```

### Test 2: Memory Recall
```
You: what do I like to drink?
Expected: Response mentioning coffee from memory or LLM context
```

### Test 3: Status
```
You: status
Expected: Session ID + memory mode (hybrid or sqlite-only)
```

### Test 4: Goal Setting
```
You: remind me to review the logs in 30 minutes
Expected: "✓ Goal set: review the logs"
```

### Test 5: LLM Question
```
You: what is 2 + 2?
Expected: LLM response
```

### Test 6: Exit
```
You: exit
Expected: Clean shutdown, no tracebacks
```

### Test 7: Persistence
1. Exit Jarvis
2. Restart: `python main.py`
3. Ask: `what do I like to drink?`
- [ ] Recalls "coffee" from the previous session

---

## RUNNING TESTS

```powershell
# Full suite
pytest -q

# CI-equivalent (requires pytest-timeout)
pytest tests/ -q --timeout=30 --tb=short

# Specific area
pytest tests/test_memory.py -v
```

Baseline: **375 passed, 5 skipped** in ~3 minutes.

---

## AUDIT LOG VERIFICATION

```powershell
python main.py --verify
```
Expected: `[OK] Audit OK - N entries verified`

---

## PRODUCTION MODE

Set in `.env` or shell before starting:
```
JARVIS_ENV=production
JARVIS_DASHBOARD_TOKEN=<strong-secret>
```

In production mode:
- Config not found → hard exit (exit code 2)
- `core.controller_v2` import failure → hard exit (no silent fallback to legacy controller)
- To allow legacy fallback explicitly: `JARVIS_ALLOW_LEGACY_CONTROLLER=1`

---

## TROUBLESHOOTING

### Ollama not reachable
```powershell
ollama serve               # start in a separate terminal
ollama list                # confirm a model is available
curl http://localhost:11434
```

### Memory DB errors
```powershell
python -c "from core.memory.hybrid_memory import HybridMemory; m = HybridMemory(); print(m.stats())"
```

### Dashboard 401 / auth issues
- Confirm `JARVIS_DASHBOARD_TOKEN` is set in `.env` and matches the `--dashboard-token` argument
- Use `X-Dashboard-Token` header for API calls
- WebSocket: connect as `ws://localhost:7070/ws?token=<your-token>`

### Legacy controller fallback in non-production
The system falls back to `core.controller` only when `core.controller_v2` fails to import.
A `WARNING` is logged. In production this is blocked unless `JARVIS_ALLOW_LEGACY_CONTROLLER=1` is set.

---

## SIGN-OFF

**Deployment Date:** ___________________  
**Deployed By:** ___________________  
**Health Check Passed?** [ ] Yes [ ] No  
**LLM Connected?** [ ] Yes [ ] No  
**Memory Persisting?** [ ] Yes [ ] No  

**DEPLOYMENT STATUS:** [ ] ✓ VERIFIED [ ] ⚠ ISSUES [ ] ✗ FAILED
