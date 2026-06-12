# 16 Deployment

## 1. System Intent (WHY does this subsystem exist?)
The deployment subsystem exists to bridge the transition from volatile source code to a durable, reproducible, and isolated execution state. Because Jarvis orchestrates complex external resources—ranging from local hardware (microphone/screen via OS integrations) to local inference engines (Ollama) and varied cloud APIs—the deployment layer ensures deterministic instantiation. It insulates the application from host OS variance, guarantees the persistence of Long-Term Memory (LTM) across reboots, and protects the system from upstream dependency regressions in the highly volatile ML ecosystem.

## 2. Core Responsibilities (WHAT responsibility does it own?)
- **Dependency Isolation & Determinism:** Manages the strict resolution of libraries through a layered dependency strategy (base, integrations, desktop, voice) and enforces a locked resolution tree (`requirements.lock`) to prevent execution drift.
- **Pre-Flight Orchestration:** Validates the necessary ambient environment parameters—such as Python version (3.11+), `.env` secret availability, and local LLM daemon (Ollama) health—before permitting the core system loop to spin up.
- **Persistence Mapping & State Safety:** Defines and guarantees the structural mount points for stateful directories (`/app/memory`, `/app/chroma_db`, `/app/logs`, `/app/workspace`) to decouple the ephemeral compute instance from durable agent data.
- **Lifecycle & CI/CD Governance:** Controls the entrypoint lifecycle (via PowerShell bootstrapping or Docker container commands), handles graceful shutdown coordinations, and enforces static analysis/quality gates (Ruff, Mypy, Pytest) on code integration.

## 3. System Interactivity (HOW does it interact with the rest of the system?)
- **Runtime Bootstrapping:** Interacts directly with `core.runtime.bootstrap` by passing environment contexts (`.env`) and configuration parameters (`jarvis.ini`). It configures standard I/O formats, fault diagnostics, and process-level signal handlers.
- **Peripheral Subsystems Integration:** The deployment scripts silently interface with the host machine to wake or spawn dependent processes (e.g., verifying Ollama listening on `localhost:11434` or injecting desktop GUI hooks).
- **Subsystem Feature Toggling:** By choosing which `requirements/*.txt` layer is installed (e.g., `desktop.txt` vs. `voice.txt`), the deployment configuration directly dictates the available capabilities of the internal registry, turning on or off computer-use/voice modes dynamically.
- **Health Diagnostics:** Exposes HTTP and CLI health-checks (`main.py --health-check`, internal `curl` polling on `port 8000`) that monitor the active heartbeat of the main orchestrator, driving container restart policies.

## 4. Failure Impact (WHAT would break if removed?)
- **Total State Amnesia:** Without explicit volume and workspace directory generation/mounting, containerized execution would wipe ChromaDB vector memory and `user_profile.json` upon every container exit, lobotomizing the agent.
- **Non-Deterministic Collapses:** Upstream updates to Hugging Face transformers, PyAutoGUI, or FastAPI would routinely break the runtime due to incompatible native bindings or API changes.
- **Silent Initialization Failures:** The application would crash inexplicably if the Ollama service was dormant. The bootstrap's job is to detect or spawn Ollama; without it, requests to the LLM would timeout and fault the state machine.
- **Host Pollution:** Missing virtual environment isolation in Windows deployments would overwrite global OS Python libraries, leading to dependency conflicts with other host software.

## 5. Reconstruction Strategy (HOW would it be rebuilt from scratch without source code?)
To recreate the deployment subsystem entirely from intent, follow these deterministic bindings:

1. **Target Matrix Definition:** Define two explicit build targets: a "Local Interactive" mode (PowerShell-based `Start.ps1`, accessing OS UI/Audio) and a "Headless Cloud" mode (Docker-based, accessible via Web UI).
2. **Layered Dependency Architecture:** Dependencies must be logically grouped and pinned:
   - `base.txt`: `pydantic==2.12.5`, `fastapi==0.136.3`, `uvicorn==0.41.0`, `requests==2.34.2`, `aiohttp==3.13.4`, `python-dotenv==1.2.2`, `jinja2==3.1.6`, `PyYAML>=6.0.3`, `python-multipart==0.0.29`, `parsedatetime==2.6`, `ollama==0.6.1`, `google-genai==1.65.0`, `chromadb==1.5.9`, `sentence-transformers==5.2.3`, `numpy==2.4.2`, `aiosqlite==0.22.1`, `ddgs==9.11.3`, `beautifulsoup4==4.14.3`
   - `desktop.txt`: `pyautogui>=0.9.54,<1.0.0`, `pygetwindow>=0.0.9,<1.0.0`, `pytesseract>=0.3.13,<1.0.0`, `pillow>=12.2,<13.0`, `pyperclip>=1.11,<2.0`, `opencv-python>=4.13,<5.0`, `plyer>=2.1,<3.0`
   - `integrations.txt`: `python-telegram-bot>=20.0,<23.0`, `PyGithub>=2.3.0,<3.0.0`, `icalendar>=7.1,<8.0`, `twilio>=9.10,<10.0`, `pyserial>=3.5,<4.0`, `duckduckgo-search>=6.0,<9.0`
   - `voice.txt`: `sounddevice>=0.5,<1.0`, `faster-whisper>=1.2,<2.0`, `SpeechRecognition>=3.14,<4.0`, `pyttsx3>=2.99,<3.0`, `edge-tts>=7.2,<8.0`, `pvporcupine>=4.0,<5.0`, `pvrecorder>=1.2,<2.0`
   - `full.txt`: Includes all above plus `pandas>=3.0,<4.0`, `fpdf2`, `pypdf>=6.12,<7.0`, `markdown>=3.10,<4.0`, `streamlit>=1.58,<2.0`. Compile these into a deterministic lockfile (`requirements.lock`).
3. **Write the Bootstrap Governor:** Develop an entrypoint script (e.g., `Start.ps1`) that strictly sequences startup. It must ensure the `.env` configuration template is hydrated with the following exact schema:
   ```env
   # Email
   EMAIL_ADDRESS=
   EMAIL_PASSWORD=
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   IMAP_HOST=imap.gmail.com
   # WhatsApp via Twilio
   TWILIO_ACCOUNT_SID=
   TWILIO_AUTH_TOKEN=
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
   # Home Assistant
   HOME_ASSISTANT_URL=http://homeassistant.local:8123
   HOME_ASSISTANT_TOKEN=
   # GitHub
   GITHUB_TOKEN=
   GITHUB_DEFAULT_REPO=
   PORCUPINE_ACCESS_KEY=
   CALENDAR_ICS_PATH=data/calendar.ics
   # Cloud LLM Fallback
   GROQ_API_KEY=
   OPENAI_API_KEY=
   ANTHROPIC_API_KEY=
   CLOUD_LLM_FALLBACK_ENABLED=true
   HF_TOKEN=
   # Web Search
   TAVILY_API_KEY=
   ```
   The bootstrap must also check Python boundaries, ping `localhost:11434/api/tags` to stall/spawn Ollama, and pre-create required directory trees (`/app/data`, `/app/memory`, `/app/chroma_db`, `/app/logs`, `/app/outputs`, `/app/workspace`, `/app/runtime`).
4. **Design the Docker Topology & Dashboard Bootstrapping:**
   Create a `Dockerfile` derived from `python:3.11-slim`. Set the environment variable `PORT=8000`.
   Map the exact explicit `VOLUME` layers: `/app/data`, `/app/memory`, `/app/chroma_db`, `/app/logs`, `/app/outputs`, `/app/workspace`, `/app/runtime`.
   Expose port `8000` (which handles the Dashboard/API).
   Define the exact internal health-check script querying the application heartbeat:
   `HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:${PORT}/health || exit 1`
   Define the entrypoint command `CMD ["python", "main.py"]`. The dashboard is a FastAPI ASGI application started via `uvicorn.run` during `main.py` bootstrapping (on port 7070 by default, or 8000 if configured via ENV).
5. **Implement CI/CD Quality Gates:** Configure a GitHub Action pipeline that pulls the repository, installs the dependency lockfile, and refuses merges that fail static type-checking (`mypy`), linting (`ruff`), or unit test suites, guaranteeing trunk stability.
