from __future__ import annotations

import asyncio
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
_controller = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MEMORY_DB = PROJECT_ROOT / "memory" / "memory.db"

app = FastAPI(title="Jarvis Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def set_controller(controller: Any) -> None:
    global _controller
    _controller = controller


def update_state(**kwargs: Any) -> None:
    """Called by any core/ module to push live state to GUI."""
    for key, value in kwargs.items():
        if hasattr(_state, key):
            setattr(_state, key, value)


def _is_authorized(request: Request) -> bool:
    expected = os.environ.get("JARVIS_DASHBOARD_TOKEN", "jarvis")
    provided = request.headers.get("X-Dashboard-Token", "")
    return bool(provided) and provided == expected


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
        return


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


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "state": _state})


@app.get("/memory")
async def memory_page(request: Request, q: str = ""):
    memories: list[dict[str, str]] = []
    message = ""

    if not MEMORY_DB.exists():
        message = "Memory database not found yet."
    else:
        try:
            search = f"%{q}%"
            with sqlite3.connect(MEMORY_DB) as conn:
                rows = conn.execute(
                    """
                    SELECT timestamp, category, content
                    FROM memories
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                    """,
                    (search,),
                ).fetchall()
            for timestamp, category, content in rows:
                memories.append(
                    {
                        "timestamp": str(timestamp or ""),
                        "category": str(category or ""),
                        "content": str(content or ""),
                    }
                )
        except sqlite3.Error:
            message = "Memory database is currently unavailable."

    if not memories and not message:
        message = "No matching memories found." if q else "No memories found yet."

    return templates.TemplateResponse(
        "memory.html",
        {"request": request, "memories": memories, "q": q, "message": message},
    )


@app.get("/goals")
async def goals_page(request: Request):
    goals: list[dict[str, Any]] = []
    message = ""
    manager = _goal_manager()

    if manager is None:
        message = "Goals available after Session 2"
    else:
        try:
            goals = _load_active_goals(manager)
            _refresh_goal_count()
            if not goals:
                message = "No active goals yet."
        except Exception:
            message = "Unable to load goals right now."

    return templates.TemplateResponse(
        "goals.html",
        {"request": request, "goals": goals, "message": message},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "state": _state.state,
        "ollama_online": _state.ollama_online,
        "memory_count": _state.memory_count,
        "active_goals": _state.active_goals,
        "uptime_seconds": round(time.time() - _state._start_time, 1),
        "model": _state.model,
    }


@app.post("/command")
async def command(request: Request):
    if not _is_authorized(request):
        return _unauthorized()

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    text = str(payload.get("text", "")).strip()
    if _controller is None:
        response_text = "Jarvis core not connected yet."
        update_state(last_input=text, last_response=response_text)
        return {"response": response_text}

    try:
        result = _controller.process(text)
        if asyncio.iscoroutine(result):
            result = await result
        response_text = str(result)
    except Exception as exc:
        response_text = f"Command failed: {exc}"

    update_state(last_input=text, last_response=response_text)
    return {"response": response_text}


@app.post("/goals/add")
async def goals_add(request: Request):
    if not _is_authorized(request):
        return _unauthorized()

    manager = _goal_manager()
    if manager is None:
        return {"error": "Goal manager not wired yet"}

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    description = str(payload.get("description", "")).strip()
    priority = int(payload.get("priority", 1) or 1)
    if not description:
        return {"error": "Description is required"}

    try:
        if hasattr(manager, "create") and callable(getattr(manager, "create")):
            manager.create(description=description, priority=priority)
        elif hasattr(manager, "create_goal") and callable(getattr(manager, "create_goal")):
            manager.create_goal(description=description, priority=priority)
        else:
            return {"error": "Goal manager not wired yet"}
        _refresh_goal_count()
        return {"ok": True}
    except Exception:
        return {"error": "Goal manager not wired yet"}


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
            return {"error": "Goal manager not wired yet"}
        _refresh_goal_count()
        return {"ok": True}
    except Exception:
        return {"error": "Goal manager not wired yet"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_ws_payload())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7070)
