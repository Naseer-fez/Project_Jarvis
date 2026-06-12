# Infrastructure Specification

## 1. Deployment Architecture
Jarvis is designed with a **Local-First Deployment Architecture**, prioritizing user privacy and local machine execution, while also offering containerized and packaged deployment options. The system supports three primary modes of deployment:
1. **Source/Virtual Environment Execution**: Direct execution on the host OS via a Python 3.11 virtual environment (`jarvis_env`).
2. **Containerized Execution**: Execution within an isolated Docker container based on Linux.
3. **Standalone Executable**: A packaged binary deployment generated using PyInstaller, bundling the runtime and application code.

### 1.1 Local Components
- **Core Engine**: A FastAPI-driven backend.
- **LLM Engine**: Locally hosted models via **Ollama**, running as an external daemon process on `localhost:11434`.
- **Vector Database**: Local instance of ChromaDB stored natively on disk (`/app/chroma_db` in Docker, `chroma_db/` locally).
- **Relational Storage**: Local SQLite database managed asynchronously via `aiosqlite`.

## 2. Environment Requirements
- **Operating System**: 
  - Natively supports Windows (via extensive PowerShell automation scripts `install.ps1`, `Start.ps1`).
  - Supports Linux via Docker (`python:3.11-slim` base image) and GitHub Actions.
- **System Dependencies**:
  - Python 3.11+ is strictly required.
  - Tesseract OCR (embedded in `bin/tesseract` or provided by the host system).
  - External daemon: Ollama (highly recommended for local inference).
- **Network Requirements**:
  - The application exposes port `8000` by default (FastAPI).
  - Expects Ollama to be reachable on `http://localhost:11434/`.

## 3. Runtime Dependencies
Dependencies are locked and managed via `requirements.lock` and categorized into feature-specific groups (`requirements/base.txt`, `full.txt`, etc.). Key runtime dependencies include:
- **Web & API Framework**: `fastapi`, `uvicorn`, `pydantic`, `starlette`.
- **LLM & AI Integrations**: `ollama`, `google-genai`, `chromadb`, `sentence-transformers`, `numpy`, `torch`.
- **Networking & Web Scraping**: `requests`, `aiohttp`, `beautifulsoup4`, `httpx`, `ddgs`.
- **Database**: `aiosqlite`.
- **Observability**: `opentelemetry` ecosystem packages.
- **Cluster Integration**: `kubernetes==36.0.0` is present in the lock file (indicating capability for cluster interaction or control, though no native project cluster orchestrations exist).

## 4. Docker & Containerization
Jarvis ships with a production-ready `Dockerfile` providing an isolated Linux environment:
- **Base Image**: `python:3.11-slim`
- **Exposed Port**: `8000`
- **Volume Mount Points**: Pre-creates directories for persistence to prevent data loss when containers are destroyed: `/app/data`, `/app/memory`, `/app/chroma_db`, `/app/logs`, `/app/outputs`, `/app/workspace`, `/app/runtime`.
- **Healthcheck**: Configured to ping `http://localhost:${PORT}/health` every 30s with a 10s timeout to ensure API responsiveness.
- **Initialization**: Installs system tools (`build-essential`, `curl`), upgrades pip, and strictly relies on `requirements.lock` for deterministic builds.

## 5. Kubernetes & Cloud Resources
- **Kubernetes**: There are **no native Kubernetes deployment manifests** (YAML files) or Helm charts provided for deploying Jarvis into a cluster. The system is designed to run as a local singleton rather than a distributed microservice. 
- **Cloud Resources & Hosting**: No Infrastructure as Code (IaC) templates (e.g., Terraform, AWS CloudFormation, or GCP Deployment Manager) are present. The architecture heavily relies on local execution and orchestration. Cloud hosting is entirely optional and left to the user, with cloud-based LLM providers acting merely as fallbacks to Ollama.

## 6. CI/CD & Build Process
### 6.1 Continuous Integration (CI)
CI is handled exclusively via **GitHub Actions workflows**:
- **Python CI (`.github/workflows/python-ci.yml`)**: Triggered on push and pull requests.
  - Provisions an `ubuntu-latest` runner.
  - Checks out the code and sets up Python 3.11 with `pip` dependency caching.
  - Installs dependencies strictly from `requirements.lock`.
  - Runs Static Analysis: Code formatting and linting via `ruff check .` and type checking via `mypy .`.
  - Executes the automated test suite: `pytest tests/ -q --timeout=30 --tb=short` injected with a `JARVIS_ENV: test` flag.
- **Daily Auto-Commit (`.github/workflows/daily_auto_commit.yml`)**: A scheduled utility workflow running cron jobs to maintain a commit streak by flushing files from a staging pool.

### 6.2 Build Process
The primary application build outputs are:
1. **Docker Image**: Built using standard `docker build -t jarvis .` commands.
2. **Standalone Executable**: Built using **PyInstaller**. The compilation configuration is strictly defined in `jarvis.spec`:
   - Bundles `main.py`, template assets (`dashboard/templates`, `dashboard/static`), workflows, plugins, and native binaries (`bin/tesseract`).
   - Hardcodes heavyweight ML dependencies (`torch`, `transformers`) as `hiddenimports`.
   - Actively strips out environment secrets (`.env` files) to ensure credentials are not hard-baked into the compiled binary bundle.

## 7. Release Process
There is no automated Continuous Deployment (CD) pipeline configured to push assets to public registries (e.g., PyPI, Docker Hub, or GitHub Releases). Releases are built manually and users are expected to consume the application either from source or via the provided installation wrappers:
- **Windows Bootstrapper**: Users execute `install.ps1`, which serves as a robust install orchestrator setting up the `jarvis_env` virtual environment, handling `pip` installations, dynamically creating `.env` configurations, verifying local Ollama availability, and running system diagnostic checks (`main.py --health-check`).
- **Local Startup**: The application is started locally via `Start.ps1`, which gracefully manages local execution by verifying the active virtual environment and automatically running Ollama as a background daemon if it is not currently active on port 11434.

## 8. Secrets Management
Secrets are managed securely but locally. There is no integration with enterprise secret managers (like AWS Secrets Manager, HashiCorp Vault, or Azure Key Vault). 
- **File Hierarchy**: Relying on `.env` (active state overrides) and `.env.example` (schema). `.env` files are correctly excluded from version control.
- **Dynamic Provisioning**: The `install.ps1` script dynamically provisions random secure keys if missing:
  - `JARVIS_SECRET_KEY`: A base64-encoded 32-byte cryptographic key using `System.Security.Cryptography.RandomNumberGenerator`.
  - `JARVIS_ADMIN_USER`: Defaults to `admin`.
  - `JARVIS_ADMIN_PASSWORD`: A securely generated 16-character randomized alphanumeric password.
  - `JARVIS_DASHBOARD_TOKEN`: A unique 16-byte token for UI authentication.
- **Build Security**: The PyInstaller pipeline script explicitly sanitizes the build by actively dropping any matched `.env` file from the final packaged output bundle.
