@echo off
SETLOCAL EnableDelayedExpansion
title JARVIS Session 5 - Automated Setup

:: ══════════════════════════════════════════════════════
::  JARVIS SETUP SCRIPT - Session 5 (Voice Layer)
::  Source : C:\Users\FEZ NASEER\Downloads\Project_Jarvis.zip
::  Dest   : D:\AI\Jarvis
::  Author : Jarvis Project
:: ══════════════════════════════════════════════════════

set "SOURCE_ZIP=C:\Users\FEZ NASEER\Downloads\Project_Jarvis.zip"
set "DEST_DIR=D:\AI\Jarvis"
set "PYTHON_EXE=python"
set "LOG_FILE=%DEST_DIR%\setup_log.txt"

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║       J.A.R.V.I.S  -  Session 5 Setup               ║
echo ║       Voice Layer: Whisper + Piper + Porcupine       ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ── STEP 0: Verify Prerequisites ─────────────────────────
echo [0/7] Checking prerequisites...

%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ollama not found. Download from: https://ollama.com
    echo           Install Ollama, then re-run this script.
    echo           Continuing setup without Ollama...
)

if not exist "%SOURCE_ZIP%" (
    echo [WARNING] ZIP not found at: %SOURCE_ZIP%
    echo           Assuming files already extracted or in current directory.
    echo           Continuing...
    goto :skip_extract
)

:: ── STEP 1: Create Directory Structure ───────────────────
echo [1/7] Creating directory structure...
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
if not exist "%DEST_DIR%\logs" mkdir "%DEST_DIR%\logs"
if not exist "%DEST_DIR%\models\piper" mkdir "%DEST_DIR%\models\piper"
if not exist "%DEST_DIR%\chroma_db" mkdir "%DEST_DIR%\chroma_db"
echo        Directories created: OK

:: ── STEP 2: Extract ZIP ──────────────────────────────────
echo [2/7] Extracting Project_Jarvis.zip...
powershell -command "Expand-Archive -Path '%SOURCE_ZIP%' -DestinationPath '%DEST_DIR%' -Force"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Extraction failed. Check that the ZIP exists and is not corrupted.
    pause
    exit /b 1
)
echo        Extraction: OK

:skip_extract

:: ── STEP 3: Virtual Environment ──────────────────────────
echo [3/7] Setting up Python virtual environment...
cd /d "%DEST_DIR%"

if exist "jarvis_env\Scripts\activate.bat" (
    echo        Virtual env already exists. Skipping creation.
) else (
    %PYTHON_EXE% -m venv jarvis_env
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Virtual environment created: OK
)

call .\jarvis_env\Scripts\activate.bat
echo        Activated: jarvis_env

:: ── STEP 4: Upgrade pip ──────────────────────────────────
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo        pip upgraded: OK

:: ── STEP 5: Install Dependencies ─────────────────────────
echo [5/7] Installing dependencies (this may take 5-15 minutes)...
echo        Installing core requirements...

pip install httpx asyncio-throttle python-dotenv --quiet
echo        Core: OK

echo        Installing memory stack (chromadb + sentence-transformers)...
pip install chromadb sentence-transformers --quiet
echo        Memory stack: OK

echo        Installing PyTorch (CPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
echo        PyTorch CPU: OK

echo        Installing Whisper STT...
pip install openai-whisper --quiet
echo        Whisper: OK

echo        Installing audio + wake word...
pip install pyaudio pvporcupine pyttsx3 numpy --quiet
echo        Audio/Wake Word: OK

echo        Installing all requirements.txt...
pip install -r requirements.txt --quiet
echo        All dependencies: OK

:: ── STEP 6: Ollama Models ────────────────────────────────
echo [6/7] Setting up Ollama models...

where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo        Starting Ollama service...
    start /b ollama serve
    timeout /t 5 /nobreak >nul

    echo        Pulling DeepSeek-R1:8b (planning model ~4.7GB)...
    echo        This will take several minutes on first run.
    ollama pull deepseek-r1:8b
    echo        DeepSeek-R1:8b: OK

    echo        Pulling llava (vision model ~4.7GB)...
    ollama pull llava
    echo        llava: OK

    echo        Pulling Whisper model via Ollama...
    ollama pull whisper
    echo        Whisper (Ollama): OK
) else (
    echo        [SKIP] Ollama not installed. Install from https://ollama.com
    echo               Then run: ollama pull deepseek-r1:8b
)

:: ── STEP 7: Environment Config ───────────────────────────
echo [7/7] Creating configuration file...
if not exist "%DEST_DIR%\.env" (
    copy "%DEST_DIR%\.env.example" "%DEST_DIR%\.env" >nul 2>&1
    echo        Created .env from template.
    echo.
    echo ┌─────────────────────────────────────────────────────┐
    echo │  ACTION REQUIRED: Edit .env and add:               │
    echo │  PORCUPINE_ACCESS_KEY=your_free_key_here           │
    echo │  Get free key at: https://console.picovoice.ai/    │
    echo └─────────────────────────────────────────────────────┘
) else (
    echo        .env already exists. Skipping.
)

:: ── Optional: Download Piper Voice Model ─────────────────
echo.
echo [OPTIONAL] Downloading Piper TTS voice model...
where curl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if not exist "%DEST_DIR%\models\piper\en_US-lessac-medium.onnx" (
        curl -L -o "%DEST_DIR%\models\piper\en_US-lessac-medium.onnx" ^
            "https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx" --silent
        curl -L -o "%DEST_DIR%\models\piper\en_US-lessac-medium.onnx.json" ^
            "https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json" --silent
        echo        Piper model downloaded: OK
    ) else (
        echo        Piper model already exists. Skipping.
    )
) else (
    echo        [SKIP] curl not found. Download Piper model manually:
    echo        https://github.com/rhasspy/piper/releases
)

:: ── Done ─────────────────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║                 SETUP COMPLETE!                      ║
echo ║                                                      ║
echo ║  Project: %DEST_DIR%
echo ║  Python:  jarvis_env (activated)                    ║
echo ║                                                      ║
echo ║  NEXT STEPS:                                         ║
echo ║  1. Edit D:\AI\Jarvis\.env                          ║
echo ║     Add: PORCUPINE_ACCESS_KEY=<your_key>            ║
echo ║  2. cd D:\AI\Jarvis                                  ║
echo ║  3. jarvis_env\Scripts\activate                     ║
echo ║  4. python main_v2.py                               ║
echo ╚══════════════════════════════════════════════════════╝
echo.
pause
