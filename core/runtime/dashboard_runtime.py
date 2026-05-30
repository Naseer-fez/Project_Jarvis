from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from typing import Any


class DashboardRuntime:
    def __init__(
        self,
        host: str,
        port: int,
        log: logging.Logger,
    ) -> None:
        self.host = host
        self.port = port
        self.log = log
        self._server = None
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None

    async def start(self, controller: Any, health_report: Any | None = None) -> None:
        if self._thread and self._thread.is_alive():
            return

        import uvicorn
        from dashboard.server import app as dashboard_app
        from dashboard.server import set_controller, update_state

        set_controller(controller)
        config = uvicorn.Config(
            dashboard_app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _serve() -> None:
            try:
                self._server.run()
            except BaseException as exc:
                self._thread_error = exc

        self._thread_error = None
        self._thread = threading.Thread(
            target=_serve,
            name="jarvis-dashboard",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._thread_error is not None:
                raise RuntimeError(
                    "Dashboard thread crashed during startup"
                ) from self._thread_error
            if getattr(self._server, "started", False):
                break
            if self._thread and not self._thread.is_alive():
                raise RuntimeError("Dashboard server exited before reporting ready")
            await asyncio.sleep(0.1)

        if not getattr(self._server, "started", False):
            self.log.warning("Dashboard startup was not confirmed within 10 seconds")

        llm_obj = getattr(controller, "llm", None)
        model_name = getattr(llm_obj, "model", getattr(llm_obj, "model_name", "unknown"))
        active_goals = 0
        goal_manager = getattr(controller, "goal_manager", None)
        if hasattr(goal_manager, "active_goals"):
            with contextlib.suppress(Exception):
                active_goals = len(goal_manager.active_goals())

        update_state(
            session_id=str(getattr(controller, "session_id", "jarvis")),
            model=str(model_name),
            state="IDLE",
            active_goals=active_goals,
            ollama_online=bool(getattr(health_report, "ollama_reachable", False)),
        )
        self.log.info("Dashboard listening on http://%s:%s", self.host, self.port)

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            try:
                self._server.should_exit = True
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.log.warning(
                    "Dashboard thread did not stop within %.1f seconds",
                    timeout,
                )

        with contextlib.suppress(Exception):
            from dashboard.server import update_state

            update_state(state="OFFLINE")


__all__ = ["DashboardRuntime"]
