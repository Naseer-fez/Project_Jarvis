# Project Requirements

This repository uses a modular approach for managing dependencies to keep the core application lightweight. Depending on the features you want to enable, you can install specific requirement files located in the `requirements/` directory.

## Default Installation

By default, the main `requirements.txt` file installs only the core runtime (equivalent to `requirements/base.txt`).

```bash
pip install -r requirements.txt
```

---

## Modular Feature Groups

To install optional features, run:
```bash
pip install -r requirements/<file_name>.txt
```

### 1. Core Runtime (`requirements/base.txt`)
These are the essential packages required to run the core application.
- **Web & API:** `pydantic`, `fastapi`, `uvicorn`, `requests`, `aiohttp`, `python-dotenv`, `jinja2`
- **LLM & Memory:** `ollama`, `google-genai`, `chromadb`, `sentence-transformers`, `numpy`
- **Web Research:** `ddgs`, `beautifulsoup4`

### 2. Voice Stack (`requirements/voice.txt`)
Includes `base.txt` plus libraries required for voice interactions, wake word detection, and Text-To-Speech (TTS).
- **Packages:** `sounddevice`, `faster-whisper`, `SpeechRecognition`, `pyttsx3`, `edge-tts`, `pvporcupine`, `pvrecorder`

### 3. Desktop Stack (`requirements/desktop.txt`)
Includes `base.txt` plus dependencies for optional desktop and computer-use integration.
- **Packages:** `pyautogui`, `pygetwindow`, `pytesseract`, `pillow`

### 4. Integrations SDKs (`requirements/integrations.txt`)
Includes `base.txt` plus optional SDKs for connecting with external services.
- **Packages:** `python-telegram-bot` (>=20.0), `PyGithub` (>=2.3.0)

### 5. Development Tooling (`requirements/dev.txt`)
Includes `base.txt` plus testing and code quality tools for developers and contributors.
- **Packages:** `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-timeout`, `mypy`, `black`, `ruff`, `types-requests`

### 6. Full Installation (`requirements/full.txt`)
If you want to install all available modules at once (Core + Voice + Desktop + Integrations + Dev), run:
```bash
pip install -r requirements/full.txt
```
