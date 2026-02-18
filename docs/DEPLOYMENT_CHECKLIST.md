# JARVIS SESSION 3 - DEPLOYMENT VERIFICATION CHECKLIST

**Version:** 0.1 Text Mode  
**Session:** 3  
**Date:** February 16, 2026

---

## FILE INTEGRITY CHECK

Verify you have all these files in `D:\AI\Jarvis\`:

```
D:\AI\Jarvis\
├── core\
│   ├── __init__.py
│   ├── controller.py
│   ├── intents.py
│   └── llm.py
├── memory\
│   ├── __init__.py
│   ├── long_term.py
│   └── short_term.py
├── main.py
├── requirements.txt
├── README.md
├── QUICK_START.md
├── SESSION_3_SUMMARY.md
└── DEPLOYMENT_CHECKLIST.md (this file)
```

**Total files:** 12 (excluding generated files)

---

## PRE-DEPLOYMENT CHECKS

- [ ] Python 3.10+ installed (`python --version`)
- [ ] D: drive has at least 10GB free space
- [ ] Administrator access available
- [ ] Internet connection active (for initial setup)

---

## INSTALLATION VERIFICATION

### Phase 1: Ollama Setup
- [ ] Ollama downloaded from https://ollama.com
- [ ] Ollama installed successfully
- [ ] `ollama --version` shows version number
- [ ] `ollama pull deepseek-r1:8b` completed (~5.2 GB)
- [ ] `ollama list` shows deepseek-r1:8b

### Phase 2: Python Environment
- [ ] Navigated to `D:\AI\Jarvis\`
- [ ] Virtual environment created (`python -m venv jarvis_env`)
- [ ] Virtual environment activated (see `(jarvis_env)` in prompt)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] No error messages during installation

### Phase 3: Ollama Server
- [ ] Separate PowerShell window opened
- [ ] `ollama serve` running (don't close this window)
- [ ] No error messages from Ollama

### Phase 4: First Run
- [ ] Jarvis started with `python main.py`
- [ ] Banner displayed correctly
- [ ] Session ID shown (8 characters)
- [ ] LLM Status shows "✓ Connected"
- [ ] Prompt shows "You: "

---

## FUNCTIONAL VERIFICATION

### Test 1: Help Command
```
You: help
Expected: List of available commands
Result: [ ] Pass [ ] Fail
```

### Test 2: Memory Storage
```
You: remember I like coffee
Expected: "✓ I'll remember that: favorite_thing = coffee"
Result: [ ] Pass [ ] Fail
```

### Test 3: Memory Storage (Different Pattern)
```
You: my name is [YourName]
Expected: "✓ I'll remember that: name = [yourname]"
Result: [ ] Pass [ ] Fail
```

### Test 4: Memory Recall
```
You: what do I like to drink?
Expected: "Based on what I know: coffee"
Result: [ ] Pass [ ] Fail
```

### Test 5: Status Command
```
You: status
Expected: System status with session info
Result: [ ] Pass [ ] Fail
```

### Test 6: LLM Question
```
You: what is 2+2?
Expected: Response from LLM (should mention 4)
Result: [ ] Pass [ ] Fail
```

### Test 7: Exit
```
You: exit
Expected: "Goodbye! Session saved." + clean shutdown
Result: [ ] Pass [ ] Fail
```

---

## PERSISTENCE VERIFICATION

### Test 8: Restart and Recall
After completing Test 7:

1. [ ] Restart Jarvis (`python main.py`)
2. [ ] Type: `what do I like to drink?`
3. [ ] Expected: Still recalls "coffee" from previous session
4. [ ] Result: [ ] Pass [ ] Fail

### Test 9: Database File Created
- [ ] File exists: `D:\AI\Jarvis\memory\memory.db`
- [ ] File size > 0 bytes
- [ ] File can be opened in SQLite browser (optional)

---

## ERROR CHECKING

### Common Issues Resolved?

- [ ] No "module not found" errors
- [ ] No "Ollama not available" warnings (if Ollama is running)
- [ ] No file permission errors
- [ ] No database write errors
- [ ] Console output is clean (no Python tracebacks)

---

## PERFORMANCE METRICS

Record your results:

- **Initial startup time:** _______ seconds
- **Response time for memory recall:** _______ seconds
- **Response time for LLM question:** _______ seconds
- **Memory.db size after 10 exchanges:** _______ KB

Expected values:
- Startup: 1-3 seconds
- Memory recall: <0.5 seconds
- LLM response: 2-10 seconds (depends on CPU)
- Database size: <100 KB initially

---

## FINAL VALIDATION

All of these must be TRUE:

- [ ] **All 9 functional tests passed**
- [ ] Memory persists across sessions
- [ ] No errors during normal operation
- [ ] LLM responds to questions
- [ ] Database file exists and grows
- [ ] Console output is clean

---

## POST-DEPLOYMENT TASKS

If all checks passed:

1. [ ] Create backup of `memory.db` (optional)
2. [ ] Bookmark this checklist for reference
3. [ ] Read `README.md` for full feature documentation
4. [ ] Review `SESSION_3_SUMMARY.md` for technical details
5. [ ] Prepare for Session 4 (voice integration)

---

## TROUBLESHOOTING REFERENCE

### If Test 6 (LLM Question) Fails:

1. Verify Ollama is running: `ollama serve`
2. Check model exists: `ollama list`
3. Test Ollama directly: `ollama run deepseek-r1:8b "What is 2+2?"`
4. Check Windows Firewall isn't blocking port 11434

### If Test 8 (Persistence) Fails:

1. Check `memory.db` exists
2. Verify write permissions on `D:\AI\Jarvis\memory\`
3. Look for error messages in console during storage
4. Try running as administrator once

### If Any Test Fails:

1. Note the exact error message
2. Check console output for Python tracebacks
3. Verify virtual environment is activated
4. Confirm all files from the checklist exist
5. Restart from Phase 2 if needed

---

## SUCCESS CRITERIA

**MINIMUM FOR SUCCESS:**
- Tests 1-5, 7-9 pass (8/9)
- Test 6 can fail if you skip LLM integration

**FULL SUCCESS:**
- All 9 tests pass
- No errors in console
- Database persists
- Clean shutdown

---

## SESSION 3 DELIVERABLES VERIFIED

- [ ] Core memory system working
- [ ] Intent classification accurate
- [ ] LLM integration functional (optional)
- [ ] Persistence verified
- [ ] Documentation complete

---

## READY FOR SESSION 4?

Session 4 will add:
- Whisper (Speech-to-Text)
- Piper (Text-to-Speech)
- Voice interaction loop

**Prerequisites before Session 4:**
- [ ] All Session 3 tests passing
- [ ] Comfortable with text mode operation
- [ ] Memory system understood
- [ ] Ready to add voice layer

---

## SIGN-OFF

**Deployment Date:** _______________________

**Deployed By:** _______________________

**All Tests Passed?** [ ] Yes [ ] No

**LLM Connected?** [ ] Yes [ ] No

**Memory Persisting?** [ ] Yes [ ] No

**Ready for Session 4?** [ ] Yes [ ] Not Yet

**Notes:**
_____________________________________________
_____________________________________________
_____________________________________________

---

**DEPLOYMENT STATUS:** [ ] ✓ VERIFIED [ ] ⚠ ISSUES [ ] ✗ FAILED

---

*Keep this checklist for reference when troubleshooting or preparing for Session 4.*
