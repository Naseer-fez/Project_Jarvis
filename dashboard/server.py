from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
import threading
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, Form, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field, StringConstraints

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN = "jarvis"
TrimmedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]

class CommandRequest(BaseModel):
    text: TrimmedText

class GoalAddRequest(BaseModel):
    description: TrimmedText
    priority: int = Field(default=1, ge=1, le=10)


@dataclass
class JarvisState:
    session_id: str = ""
    state: str = "OFFLINE"
    last_input: str = ""
    last_response: str = ""
    model: str = "unknown"
    memory_count: int = 0
    active_goals: int = 0
    ollama_online: bool = False
    _start_time: float = field(default_factory=time.time)



_state = JarvisState()
_state_lock = threading.Lock()
_controller = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def _resolve_memory_db() -> Path:
    """Return the SQLite DB path from env → config default → hard default."""
    env_val = os.environ.get("JARVIS_MEMORY_DB", "")
    if env_val:
        p = Path(env_val)
        return p if p.is_absolute() else PROJECT_ROOT / p
    # Mirror jarvis.ini [memory] sqlite_file = data/jarvis_memory.db
    config = getattr(_controller, "config", None)
    if config:
        try:
            raw = config.get("memory", "sqlite_file", fallback="data/jarvis_memory.db")
            p = Path(raw)
            return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    return PROJECT_ROOT / "data" / "jarvis_memory.db"



@asynccontextmanager
async def lifespan(app: FastAPI):
    _warn_default_token()
    await asyncio.to_thread(_get_auth_manager)
    yield
    _clicker.stop()


app = FastAPI(title="Jarvis Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

GUI_AUDIT_DIR = PROJECT_ROOT / "outputs" / "gui_audit"
GUI_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/gui-audit", StaticFiles(directory=str(GUI_AUDIT_DIR)), name="gui-audit")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def set_controller(controller: Any) -> None:
    global _controller
    _controller = controller


def update_state(**kwargs: Any) -> None:
    """Called by any core/ module to push live state to GUI."""
    with _state_lock:
        for key, value in kwargs.items():
            if hasattr(_state, key):
                setattr(_state, key, value)


def _warn_default_token() -> None:
    """Emit a loud warning when the dashboard token is the insecure default."""
    token = os.environ.get("JARVIS_DASHBOARD_TOKEN", _DEFAULT_TOKEN)
    env = os.environ.get("JARVIS_ENV", "development").lower()
    if token == _DEFAULT_TOKEN:
        if env == "production":
            logger.critical("SECURITY: Cannot start in production with default dashboard token.")
            import sys
            sys.exit(1)
        logger.warning(
            "SECURITY: JARVIS_DASHBOARD_TOKEN is set to the insecure default '%s'. "
            "Set a strong secret via the env var or --dashboard-token before exposing "
            "the dashboard on any non-loopback interface.",
            _DEFAULT_TOKEN,
        )


_auth_manager = None

def _get_auth_manager() -> Any:
    global _auth_manager
    if _auth_manager is not None:
        return _auth_manager
    from core.security.auth import AuthManager, auth_db_from_config
    config = getattr(_controller, "config", None)
    db_path = auth_db_from_config(config) if config else PROJECT_ROOT / "data" / "auth.db"
    secret_key = os.environ.get("JARVIS_SECRET_KEY")
    if not secret_key and config:
        try:
            secret_key = config.get("security", "secret_key", fallback=None)
        except Exception:
            pass
    _auth_manager = AuthManager(db_path=db_path, secret_key=secret_key)
    # Also bootstrap admin user if environment variables are set and database is empty
    _auth_manager.bootstrap_admin_from_env()
    return _auth_manager


async def _is_authorized(request: Request) -> bool:
    """Check if the request has a valid session cookie or matching token header/API token."""
    # 1. Check for valid session cookie
    session_token = request.cookies.get("jarvis_session", "")
    if session_token:
        try:
            auth = _get_auth_manager()
            user = auth.verify_session(session_token)
            if user is not None:
                return True
        except Exception:
            logger.debug("Failed verifying session cookie", exc_info=True)

    # 2. Fallback to API token or static token in headers
    provided = request.headers.get("X-Dashboard-Token", "")
    if provided:
        expected = os.environ.get("JARVIS_DASHBOARD_TOKEN", _DEFAULT_TOKEN)
        try:
            auth = _get_auth_manager()
            if await asyncio.to_thread(auth.verify_api_token, provided) or hmac.compare_digest(provided.encode(), expected.encode()):
                return True
        except Exception:
            if hmac.compare_digest(provided.encode(), expected.encode()):
                return True
    return False


async def _ws_is_authorized(websocket: WebSocket, token: str) -> bool:
    """Validate a WebSocket connection by query-parameter token or cookie."""
    # 1. Check query parameter token
    if token:
        try:
            auth = _get_auth_manager()
            user = auth.verify_session(token)
            if user is not None:
                return True
            expected = os.environ.get("JARVIS_DASHBOARD_TOKEN", _DEFAULT_TOKEN)
            if hmac.compare_digest(token.encode(), expected.encode()):
                return True
        except Exception:
            expected = os.environ.get("JARVIS_DASHBOARD_TOKEN", _DEFAULT_TOKEN)
            if hmac.compare_digest(token.encode(), expected.encode()):
                return True

    # 2. Check jarvis_session cookie
    session_token = websocket.cookies.get("jarvis_session", "")
    if session_token:
        try:
            auth = _get_auth_manager()
            user = auth.verify_session(session_token)
            if user is not None:
                return True
        except Exception:
            pass

    return False


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "Unauthorized"})


def _goal_manager() -> Any | None:
    if _controller is None or not hasattr(_controller, "goal_manager"):
        return None
    return getattr(_controller, "goal_manager")


def _format_created(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ", timespec="seconds")
        except TypeError:
            return value.isoformat()
    return str(value)


def _serialize_goal(goal: Any) -> dict[str, Any]:
    if isinstance(goal, dict):
        status = goal.get("status", "active")
        if hasattr(status, "value"):
            status = status.value
        return {
            "goal_id": str(goal.get("goal_id") or goal.get("id") or ""),
            "priority": goal.get("priority", 1),
            "description": str(goal.get("description", "")),
            "created": _format_created(goal.get("created_at") or goal.get("created")),
            "status": str(status),
        }

    status = getattr(goal, "status", "active")
    if hasattr(status, "value"):
        status = status.value
    return {
        "goal_id": str(getattr(goal, "goal_id", getattr(goal, "id", ""))),
        "priority": getattr(goal, "priority", 1),
        "description": str(getattr(goal, "description", "")),
        "created": _format_created(getattr(goal, "created_at", getattr(goal, "created", None))),
        "status": str(status),
    }


def _load_active_goals(manager: Any) -> list[dict[str, Any]]:
    if manager is None:
        return []

    if hasattr(manager, "active_goals") and callable(getattr(manager, "active_goals")):
        raw_goals = list(manager.active_goals())
    else:
        raw_goals = []
    return [_serialize_goal(goal) for goal in raw_goals]


def _refresh_goal_count() -> None:
    manager = _goal_manager()
    if manager is None:
        return
    try:
        update_state(active_goals=len(_load_active_goals(manager)))
    except Exception:
        logger.debug("Goal count refresh failed", exc_info=True)


def _load_ai_os_overview() -> dict[str, Any]:
    config = getattr(_controller, "config", None)
    overview: dict[str, Any] = {
        "blueprint": {},
        "plugins": {"count": 0, "plugins": [], "errors": {}},
        "workflows": {"count": 0, "templates": [], "errors": {}},
        "errors": [],
    }

    try:
        from core.ai_os import load_blueprint

        overview["blueprint"] = load_blueprint(config=config).summary()
    except Exception as exc:
        logger.debug("Failed loading AI OS blueprint", exc_info=True)
        overview["errors"].append(f"Blueprint unavailable: {exc}")

    try:
        from core.plugins import PluginCatalog

        overview["plugins"] = PluginCatalog(config=config).summary()
    except Exception as exc:
        logger.debug("Failed loading plugin catalog", exc_info=True)
        overview["errors"].append(f"Plugin catalog unavailable: {exc}")

    overview["workflows"] = {"count": 0, "templates": [], "errors": {}}

    return overview


def _ws_payload() -> dict[str, Any]:
    return {
        "state": _state.state,
        "last_response": _state.last_response,
        "last_input": _state.last_input,
        "session_id": _state.session_id,
        "memory_count": _state.memory_count,
        "active_goals": _state.active_goals,
        "ollama_online": _state.ollama_online,
        "uptime_seconds": round(time.time() - _state._start_time, 1),
        "model": _state.model,
    }


# ---------------------------------------------------------------------------
# Public readiness probe — intentionally unauthenticated and non-verbose.
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    """Lightweight readiness probe. Intentionally unauthenticated."""
    from core.introspection.health import run_lightweight_health_check
    import dataclasses
    import asyncio
    import time
    
    config = getattr(_controller, "config", None) if _controller else None
    
    def _get_report():
        return run_lightweight_health_check(config)
        
    report = await asyncio.to_thread(_get_report)
    
    return {
        "ok": report.is_healthy,
        "state": _state.state,
        "uptime_seconds": round(time.time() - _state._start_time, 1),
        "report": dataclasses.asdict(report),
    }


# ---------------------------------------------------------------------------
# Protected HTML pages — require X-Dashboard-Token header.
# ---------------------------------------------------------------------------

@app.get("/login")
async def login_get(request: Request):
    if await _is_authorized(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    try:
        auth = _get_auth_manager()
        user = await asyncio.to_thread(auth.authenticate, username, password)
        if user is not None:
            token = auth.sign_session(user)
            response = RedirectResponse(url="/", status_code=303)
            response.set_cookie(
                key="jarvis_session",
                value=token,
                httponly=True,
                samesite="lax",
                secure=False,
            )
            return response
    except Exception as exc:
        logger.exception("Login authentication crashed")
        return templates.TemplateResponse(request, "login.html", {"error": f"Internal system error: {exc}"})
    
    return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"})

@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("jarvis_session")
    return response


@app.get("/")
async def index(request: Request):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "index.html", {"state": _state})


@app.get("/memory")
async def memory_page(request: Request, q: str = ""):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)

    memories: list[dict[str, str]] = []
    message = ""
    memory_db = _resolve_memory_db()

    if not memory_db.exists():
        message = "Memory database not found yet."
    else:
        try:
            search = f"%{q}%"
            def _query():
                with sqlite3.connect(memory_db) as conn:
                    conn.row_factory = sqlite3.Row
                    # The real schema has preferences / episodes / conversations.
                    # We UNION them into a common (timestamp, category, content) shape
                    # so the template keeps working unchanged.
                    return conn.execute(
                        """
                        SELECT updated_at AS timestamp, 'preference' AS category, (key || ': ' || value) AS content
                        FROM preferences
                        WHERE (key || ': ' || value) LIKE ?
                        UNION ALL
                        SELECT timestamp, category, event AS content
                        FROM episodes
                        WHERE event LIKE ?
                        UNION ALL
                        SELECT timestamp, 'conversation' AS category,
                               ('User: ' || user_input || ' | Jarvis: ' || assistant_response) AS content
                        FROM conversations
                        WHERE ('User: ' || user_input || ' | Jarvis: ' || assistant_response) LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT 100
                        """,
                        (search, search, search),
                    ).fetchall()

            rows = await asyncio.to_thread(_query)
            for row in rows:
                memories.append(
                    {
                        "timestamp": str(row["timestamp"] or ""),
                        "category": str(row["category"] or ""),
                        "content": str(row["content"] or ""),
                    }
                )
        except sqlite3.Error:
            logger.exception("Memory DB query failed (db=%s)", memory_db)
            message = "Memory database is currently unavailable."

    if not memories and not message:
        message = "No matching memories found." if q else "No memories found yet."

    return templates.TemplateResponse(
        "memory.html",
        {"request": request, "memories": memories, "q": q, "message": message},
    )


@app.get("/goals")
async def goals_page(request: Request):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)

    goals: list[dict[str, Any]] = []
    message = ""
    manager = _goal_manager()

    if manager is None:
        message = "Goals available after controller start"
    else:
        try:
            goals = _load_active_goals(manager)
            _refresh_goal_count()
            if not goals:
                message = "No active goals yet."
        except Exception:
            logger.exception("goals_page: failed to load goals")
            message = "Unable to load goals right now."

    return templates.TemplateResponse(
        request,
        "goals.html",
        {"goals": goals, "message": message},
    )


@app.get("/search")
async def search_page(request: Request):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "search.html")


@app.get("/converter")
async def converter_page(request: Request):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "converter.html")


# ---------------------------------------------------------------------------
# Auto-Clicker Manager & Endpoints
# ---------------------------------------------------------------------------

class ClickerStartRequest(BaseModel):
    target: str = Field(..., min_length=1, max_length=100)
    interval: float = Field(default=5.0, ge=0.5, le=3600.0)
    continuous: bool = Field(default=False)
    min_confidence: float = Field(default=0.2, ge=0.0, le=1.0)


class AutoClickerManager:
    def __init__(self):
        self.running = False
        self.target = ""
        self.interval = 5.0
        self.continuous = False
        self.min_confidence = 0.2
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.logs: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._start_time: float = 0.0

    def add_log(self, message: str, level: str = "info"):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
        if len(self.logs) > 200:
            self.logs.pop(0)

    def start(self, target: str, interval: float, continuous: bool, min_confidence: float):
        if self.running:
            self.stop()
            
        self.running = True
        self.target = target
        self.interval = interval
        self.continuous = continuous
        self.min_confidence = min_confidence
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self._start_time = time.time()
        self.add_log(f"Auto-Clicker initialized for target: '{target}'", "info")
        
        self._task = asyncio.create_task(self._run_loop())

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.add_log("Auto-Clicker stopped.", "warning")

    async def _run_loop(self):
        from core.tools.gui_control import click_screen_target
        
        while self.running:
            self.attempts += 1
            self.add_log(f"Attempt {self.attempts}: Searching for target...", "info")
            try:
                # Run the actual click_screen_target tool
                result = await click_screen_target(
                    target=self.target,
                    occurrence=1,
                    button="left",
                    match_mode="contains",
                    min_confidence=self.min_confidence,
                )
                
                if result.success:
                    self.successes += 1
                    self.add_log(f"Click SUCCESS: Target clicked! {result.data.get('reason', '')}", "success")
                    if not self.continuous:
                        self.add_log("Continuous mode disabled. Stopping clicker.", "info")
                        self.running = False
                        break
                else:
                    self.failures += 1
                    self.add_log(f"Click FAILED: {result.error}", "error")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.failures += 1
                self.add_log(f"Error in clicker loop: {e}", "error")
            
            await asyncio.sleep(self.interval)

_clicker = AutoClickerManager()


@app.get("/clicker")
async def clicker_page(request: Request):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "clicker.html", {"clicker": _clicker})


@app.get("/api/clicker/state")
async def api_clicker_state(request: Request):
    if not await _is_authorized(request):
        return _unauthorized()
    
    uptime = 0.0
    if _clicker.running and _clicker._start_time > 0:
        uptime = round(time.time() - _clicker._start_time, 1)
        
    return {
        "running": _clicker.running,
        "target": _clicker.target,
        "interval": _clicker.interval,
        "continuous": _clicker.continuous,
        "min_confidence": _clicker.min_confidence,
        "attempts": _clicker.attempts,
        "successes": _clicker.successes,
        "failures": _clicker.failures,
        "uptime": uptime,
        "logs": _clicker.logs
    }


@app.post("/api/clicker/start")
async def api_clicker_start(request: Request, body: ClickerStartRequest):
    if not await _is_authorized(request):
        return _unauthorized()
    
    _clicker.start(
        target=body.target,
        interval=body.interval,
        continuous=body.continuous,
        min_confidence=body.min_confidence
    )
    return {"ok": True}


@app.post("/api/clicker/stop")
async def api_clicker_stop(request: Request):
    if not await _is_authorized(request):
        return _unauthorized()
    
    _clicker.stop()
    return {"ok": True}


@app.post("/api/clicker/clear-logs")
async def api_clicker_clear_logs(request: Request):
    if not await _is_authorized(request):
        return _unauthorized()
    
    _clicker.logs = []
    return {"ok": True}


@app.get("/api/clicker/screenshots")
async def api_clicker_screenshots(request: Request):
    if not await _is_authorized(request):
        return _unauthorized()
        
    gui_audit_dir = PROJECT_ROOT / "outputs" / "gui_audit"
    if not gui_audit_dir.exists():
        return []
        
    try:
        files = sorted(
            gui_audit_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return [
            {
                "name": p.name,
                "url": f"/gui-audit/{p.name}",
                "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))
            }
            for p in files[:8]
        ]
    except Exception as e:
        logger.error("Error reading screenshots: %s", e)
        return []


@app.get("/ai-os")
async def ai_os_page(request: Request):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "ai_os.html",
        {"overview": _load_ai_os_overview()},
    )


@app.get("/api/ai-os")
async def api_ai_os(request: Request):
    if not await _is_authorized(request):
        return _unauthorized()
    return _load_ai_os_overview()


# ---------------------------------------------------------------------------
# Authenticated write endpoints
# ---------------------------------------------------------------------------

@app.post("/command")
async def command(request: Request, body: CommandRequest):
    if not await _is_authorized(request):
        return _unauthorized()

    text = body.text
    import uuid
    trace_id = uuid.uuid4().hex[:8]
    logger.info("Processing command: %r", text, extra={"trace_id": trace_id})

    if _controller is None:
        response_text = "Jarvis core not connected yet."
        update_state(last_input=text, last_response=response_text)
        return {"response": response_text}

    try:
        result = _controller.process(text, trace_id=trace_id)
        if asyncio.iscoroutine(result):
            result = await result
        response_text = str(result)
    except Exception:
        logger.exception("command endpoint: process failed for input=%r", text)
        response_text = "Command failed — see server logs for details."

    update_state(last_input=text, last_response=response_text)
    return {"response": response_text}


@app.post("/goals/add")
async def goals_add(request: Request, body: GoalAddRequest):
    if not await _is_authorized(request):
        return _unauthorized()

    manager = _goal_manager()
    if manager is None:
        return {"error": "Goal manager not wired yet"}

    description = body.description
    priority = body.priority

    try:
        if hasattr(manager, "create") and callable(getattr(manager, "create")):
            manager.create(description=description, priority=priority)
        elif hasattr(manager, "create_goal") and callable(getattr(manager, "create_goal")):
            manager.create_goal(description=description, priority=priority)
        else:
            return {"error": "Goal manager does not support creation"}
        _refresh_goal_count()
        return {"ok": True}
    except Exception:
        logger.exception("goals_add: failed to create goal description=%r", description)
        return {"error": "Failed to add goal — see server logs for details."}


@app.post("/goals/complete/{goal_id}")
async def goals_complete(goal_id: str, request: Request):
    if not await _is_authorized(request):
        return _unauthorized()

    manager = _goal_manager()
    if manager is None:
        return {"error": "Goal manager not wired yet"}

    try:
        if hasattr(manager, "complete") and callable(getattr(manager, "complete")):
            manager.complete(goal_id)
        elif hasattr(manager, "complete_goal") and callable(getattr(manager, "complete_goal")):
            manager.complete_goal(goal_id)
        else:
            return {"error": "Goal manager does not support completion"}
        _refresh_goal_count()
        return {"ok": True}
    except Exception:
        logger.exception("goals_complete: failed to complete goal_id=%r", goal_id)
        return {"error": "Failed to complete goal — see server logs for details."}


class SearchRequest(BaseModel):
    path: str = "all"
    query: str = ""
    content: str = ""
    threads: int = 8
    case_sensitive: bool = False
    no_skip: bool = False


@app.post("/api/search")
async def api_search(request: Request, body: SearchRequest):
    if not await _is_authorized(request):
        return _unauthorized()
    
    from core.tools.fast_search_tool import run_fast_search
    try:
        results = await run_fast_search(
            path=body.path,
            query=body.query,
            content=body.content,
            threads=body.threads,
            case_sensitive=body.case_sensitive,
            no_skip=body.no_skip
        )
        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/convert")
async def api_convert(
    request: Request,
    file: UploadFile = File(...),
    target_format: str = Form(...)
):
    if not await _is_authorized(request):
        return _unauthorized()
        
    from core.tools.universal_converter import perform_conversion
    import tempfile
    import shutil
    
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_src:
        shutil.copyfileobj(file.file, tmp_src)
        tmp_src_path = tmp_src.name
        
    target_format_clean = target_format.strip().lower()
    try:
        output_path = perform_conversion(tmp_src_path, target_format_clean)
        base_name = Path(file.filename).stem
        out_filename = f"{base_name}.{target_format_clean.lstrip('.')}"
        
        from starlette.background import BackgroundTasks
        background_tasks = BackgroundTasks()
        background_tasks.add_task(os.unlink, tmp_src_path)
        background_tasks.add_task(os.unlink, output_path)
        
        return FileResponse(
            path=output_path,
            filename=out_filename,
            media_type="application/octet-stream",
            background=background_tasks
        )
    except Exception as e:
        if os.path.exists(tmp_src_path):
            try:
                os.unlink(tmp_src_path)
            except OSError:
                pass
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/view-file")
async def api_view_file(request: Request, path: str):
    if not await _is_authorized(request):
        return RedirectResponse(url="/login", status_code=303)
        
    try:
        file_path = Path(path).resolve()
        # Ensure target file is inside PROJECT_ROOT (prevent traversal)
        file_path.relative_to(PROJECT_ROOT)
    except (ValueError, RuntimeError):
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied. Paths must be inside project root."}
        )
        
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
        
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream"
    )


# ---------------------------------------------------------------------------
# WebSocket — token via query parameter (WS APIs don't support custom headers)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    if not await _ws_is_authorized(websocket, token):
        # 1008 = Policy Violation — standard close code for auth failures
        await websocket.close(code=1008)
        logger.warning("WebSocket connection rejected — missing or invalid token (client=%s)", websocket.client)
        return

    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_ws_payload())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("WebSocket error during stream", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Startup event — warn on insecure default token
# ---------------------------------------------------------------------------

# Startup and Shutdown are handled via the lifespan context manager above.


if __name__ == "__main__":
    _host = os.environ.get("JARVIS_DASHBOARD_HOST", "127.0.0.1")
    try:
        _port = int(os.environ.get("JARVIS_DASHBOARD_PORT", "7070"))
    except ValueError:
        _port = 7070
    uvicorn.run(app, host=_host, port=_port)
