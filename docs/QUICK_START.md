# JARVIS - QUICK START DEPLOYMENT GUIDE

**For:** Windows 10/11  
**Session:** 3 - Text Mode  
**Estimated Setup Time:** 15 minutes

---

## PRE-FLIGHT CHECKLIST

Before you begin, ensure you have:

- [ ] Windows 10/11 with administrator access
- [ ] At least 10GB free space on D: drive
- [ ] Internet connection (for initial setup only)
- [ ] Python 3.10 or higher installed

---

## STEP-BY-STEP DEPLOYMENT

### STEP 1: Download and Extract Project Files

1. Download all files from this session
2. Extract to: `D:\AI\Jarvis\`
3. Verify directory structure matches:

```
D:\AI\Jarvis\
├── core\
│   ├── __init__.py
│   ├── llm.py
│   ├── intents.py
│   └── controller.py
├── memory\
│   ├── __init__.py
│   ├── long_term.py
│   └── short_term.py
├── main.py
├── requirements.txt
├── README.md
└── SESSION_3_SUMMARY.md
```

---

### STEP 2: Install Ollama

1. **Download:** https://ollama.com
2. **Install:** Run the installer
3. **Verify:**
   ```powershell
   ollama --version
   ```
4. **Pull Model:**
   ```powershell
   ollama pull deepseek-r1:8b
   ```
   *(This downloads ~5.2 GB)*

---

### STEP 3: Create Python Virtual Environment

Open PowerShell as **Administrator**:

```powershell
# Navigate to project
cd D:\AI\Jarvis

# Create virtual environment
python -m venv jarvis_env

# Activate environment
.\jarvis_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Note:** If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### STEP 4: Start Ollama Server

Open a **new** PowerShell window:

```powershell
ollama serve
```

**Leave this window open** while using Jarvis.

---

### STEP 5: Run Jarvis

In your **original** PowerShell window:

```powershell
cd D:\AI\Jarvis
.\jarvis_env\Scripts\Activate.ps1
python main.py
```

You should see:

```
╔════════════════════════════════════════════╗
║                                            ║
║           JARVIS v0.1 - OFFLINE            ║
║      Personal AI Assistant (Text Mode)     ║
║                                            ║
╚════════════════════════════════════════════╝

Session ID: xxxxxxxx
LLM Status: ✓ Connected

Ready. How can I help?
```

---

## FIRST RUN TEST SEQUENCE

Try these commands to verify everything works:

```
You: help
[Should show command list]

You: remember I like coffee
[Should confirm storage]

You: my name is [Your Name]
[Should confirm storage]

You: what do I like to drink?
[Should recall "coffee"]

You: status
[Should show session info]

You: exit
[Should save and quit]
```

**Then restart Jarvis and type:**

```
You: what do I like to drink?
[Should STILL recall "coffee" - proving persistence works]
```

---

## TROUBLESHOOTING

### Error: "LLM not available"

**Solution:**
1. Check Ollama is running: `ollama serve` in separate terminal
2. Verify model exists: `ollama list`
3. Check port 11434 is free: `netstat -an | findstr 11434`

---

### Error: "Module not found"

**Solution:**
1. Ensure virtual environment is activated
2. Reinstall: `pip install -r requirements.txt`
3. Check Python version: `python --version` (must be 3.10+)

---

### Error: "Cannot activate virtual environment"

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Memory not persisting

**Solution:**
1. Check `D:\AI\Jarvis\memory\memory.db` exists after first run
2. Verify no permission errors in console
3. Try running as administrator once

---

## DAILY USAGE WORKFLOW

### Starting Jarvis
```powershell
# Terminal 1 (Ollama)
ollama serve

# Terminal 2 (Jarvis)
cd D:\AI\Jarvis
.\jarvis_env\Scripts\Activate.ps1
python main.py
```

### Stopping Jarvis
```
You: exit
```

Then close both PowerShell windows.

---

## DATA LOCATIONS

| What | Where |
|------|-------|
| Project code | `D:\AI\Jarvis\` |
| Virtual env | `D:\AI\Jarvis\jarvis_env\` |
| Memory database | `D:\AI\Jarvis\memory\memory.db` |
| Session logs | `D:\AI\Jarvis\data\logs\` (future) |

**IMPORTANT:** All data is local. Nothing is sent to the cloud.

---

## WHAT TO EXPECT (TEXT MODE)

### ✓ Working Features
- Memory storage and recall
- Question answering (via Ollama)
- Session persistence
- Conversation history
- Basic commands

### ✗ Not Yet Implemented
- Voice interaction (Session 4)
- File/app launching
- Semantic search
- Wake word detection

---

## NEXT STEPS

After confirming text mode works:

1. **Test thoroughly** - Try various memory patterns
2. **Report issues** - Note any errors or unexpected behavior
3. **Session 4** - We'll add voice capabilities (Whisper + Piper)

---

## VERIFY SUCCESSFUL DEPLOYMENT

Run this checklist after setup:

- [ ] Ollama installed and serving
- [ ] Python virtual environment created
- [ ] Dependencies installed (no errors)
- [ ] Jarvis starts without errors
- [ ] LLM status shows "Connected"
- [ ] Memory storage works
- [ ] Memory persists after restart
- [ ] No errors in console

**If all checked:** ✓ Deployment successful!

---

## GETTING HELP

### Console shows errors?
1. Copy the full error message
2. Check which file/line it references
3. Verify that file exists in correct location

### Ollama not connecting?
1. Check Windows Firewall
2. Verify port 11434 is not blocked
3. Try: `ollama list` to verify installation

### Dependencies failing?
1. Update pip: `python -m pip install --upgrade pip`
2. Retry: `pip install -r requirements.txt`
3. Check Python version: `python --version`

---

## IMPORTANT NOTES

1. **Internet Required:** Only for initial setup (downloading Ollama model and Python packages)
2. **Offline Operation:** After setup, Jarvis works 100% offline
3. **Memory Persistence:** Your data is stored in `memory.db` - back it up if needed
4. **Session IDs:** Each run gets a unique session ID for tracking
5. **D: Drive Policy:** Keep all data on D: to avoid filling C:

---

## SUCCESS CRITERIA

You've successfully deployed Jarvis if:

✓ Jarvis starts without errors  
✓ You can store preferences  
✓ Preferences persist after restart  
✓ LLM responds to questions  
✓ Memory database is created  
✓ Session logging works  

---

**Ready to proceed?**

Start with STEP 1 and work through sequentially.  
Each step should take 2-5 minutes.  
Total setup time: ~15 minutes.

**JARVIS v0.1 - Text Mode - Ready for Deployment**
