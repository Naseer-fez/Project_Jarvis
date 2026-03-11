from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN = "jarvis"

class CommandRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, strip_whitespace=True)

class GoalAddRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=4096, strip_whitespace=True)
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


import threading
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
    return PROJECT_ROOT / "data" / "jarvis_memory.db"


app = FastAPI(title="Jarvis Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
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


def _is_authorized(request: Request) -> bool:
    """Constant-time token comparison against the configured secret."""
    expected = os.environ.get("JARVIS_DASHBOARD_TOKEN", _DEFAULT_TOKEN)
    provided = request.headers.get("X-Dashboard-Token", "")
    if not provided:
        return False
    # hmac.compare_digest prevents timing-oracle attacks
    return hmac.compare_digest(provided.encode(), expected.encode())


def _ws_is_authorized(token: str) -> bool:
    """Validate a WebSocket query-parameter token."""
    if not token:
        return False
    expected = os.environ.get("JARVIS_DASHBOARD_TOKEN", _DEFAULT_TOKEN)
    return hmac.compare_digest(token.encode(), expected.encode())


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
    overall = "ok" if _state.state not in ("OFFLINE", "ERROR") else "degraded"
    return {
        "ok": overall == "ok",
        "state": _state.state,
        "uptime_seconds": round(time.time() - _state._start_time, 1),
    }


# ---------------------------------------------------------------------------
# Protected HTML pages — require X-Dashboard-Token header.
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    if not _is_authorized(request):
        return _unauthorized()
    return templates.TemplateResponse("index.html", {"request": request, "state": _state})


@app.get("/memory")
async def memory_page(request: Request, q: str = ""):
    if not _is_authorized(request):
        return _unauthorized()

    memories: list[dict[str, str]] = []
    message = ""
    memory_db = _resolve_memory_db()

    if not memory_db.exists():
        message = "Memory database not found yet."
    else:
        try:
            search = f"%{q}%"
            with sqlite3.connect(memory_db) as conn:
                conn.row_factory = sqlite3.Row
                # The real schema has preferences / episodes / conversations.
                # We UNION them into a common (timestamp, category, content) shape
                # so the template keeps working unchanged.
                rows = conn.execute(
                    """
                    SELECT updated_at AS timestamp, 'preference' AS category, (key || ': ' || value) AS content
                    FROM preferences
                    WHERE content LIKE ?
                    UNION ALL
                    SELECT timestamp, category, event AS content
                    FROM episodes
                    WHERE content LIKE ?
                    UNION ALL
                    SELECT timestamp, 'conversation' AS category,
                           ('User: ' || user_input || ' | Jarvis: ' || assistant_response) AS content
                    FROM conversations
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                    """,
                    (search, search, search),
                ).fetchall()
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
    if not _is_authorized(request):
        return _unauthorized()

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
        "goals.html",
        {"request": request, "goals": goals, "message": message},
    )


# ---------------------------------------------------------------------------
# Authenticated write endpoints
# ---------------------------------------------------------------------------

@app.post("/command")
async def command(request: Request, body: CommandRequest):
    if not _is_authorized(request):
        return _unauthorized()

    text = body.text
    import uuid
    trace_id = uuid.uuid4().hex[:8]
    logger.info("[trace=%s] Processing command: %r", trace_id, text)

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
    if not _is_authorized(request):
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
    if not _is_authorized(request):
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


# ---------------------------------------------------------------------------
# WebSocket — token via query parameter (WS APIs don't support custom headers)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    if not _ws_is_authorized(token):
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

@app.on_event("startup")
async def _on_startup() -> None:
    _warn_default_token()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7070)
