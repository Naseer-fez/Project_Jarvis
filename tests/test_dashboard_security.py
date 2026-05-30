import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

from dashboard.server import app, set_controller
import dashboard.server as server
from core.security.auth import AuthManager


@pytest.fixture()
def test_auth_db(tmp_path):
    db_file = tmp_path / "test_auth.db"
    auth = AuthManager(db_path=db_file, secret_key="test-signing-secret-123456789012")
    auth.create_user("admin_user", "securepassword123", is_admin=True)
    return auth, db_file


@pytest.fixture()
def mock_dashboard_controller():
    ctrl = MagicMock()
    ctrl.session_id = "test-session-id"
    ctrl.process = MagicMock(return_value="processed ok")
    ctrl.config = None
    return ctrl


def test_unauthorized_view_file_redirects():
    """Verify unauthorized calls to /api/view-file redirect to login page."""
    server._auth_manager = None
    client = TestClient(app)
    response = client.get("/api/view-file?path=config/jarvis.ini", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_authorized_view_file_success(test_auth_db, mock_dashboard_controller, tmp_path):
    """Verify authorized path viewing works for files inside project root."""
    auth, _ = test_auth_db
    server._auth_manager = auth
    set_controller(mock_dashboard_controller)

    client = TestClient(app)
    user = auth.authenticate("admin_user", "securepassword123")
    valid_session = auth.sign_session(user)

    client.cookies.set("jarvis_session", valid_session)

    # Let's create a test file inside the project root
    test_file_path = server.PROJECT_ROOT / "test_temp_doc.txt"
    test_file_path.write_text("Hello Jarvis secure area", encoding="utf-8")

    try:
        response = client.get(f"/api/view-file?path={test_file_path}")
        assert response.status_code == 200
        assert response.text == "Hello Jarvis secure area"
    finally:
        if test_file_path.exists():
            test_file_path.unlink()


def test_authorized_view_file_traversal_blocked(test_auth_db, mock_dashboard_controller, tmp_path):
    """Verify authorized path viewing blocks path traversal outside PROJECT_ROOT."""
    auth, _ = test_auth_db
    server._auth_manager = auth
    set_controller(mock_dashboard_controller)

    client = TestClient(app)
    user = auth.authenticate("admin_user", "securepassword123")
    valid_session = auth.sign_session(user)

    client.cookies.set("jarvis_session", valid_session)

    # File outside PROJECT_ROOT
    outside_file = tmp_path / "outside_secret.txt"
    outside_file.write_text("classified document", encoding="utf-8")

    # Path traversal with relative parent dirs or absolute outside paths
    paths_to_test = [
        str(outside_file),
        f"../../../../../../{outside_file.name}",
    ]

    for test_path in paths_to_test:
        response = client.get(f"/api/view-file?path={test_path}")
        assert response.status_code == 403
        assert "Access denied" in response.json().get("error", "")


def test_dashboard_lifespan(monkeypatch):
    """Verify that dashboard lifespan startup and shutdown hooks trigger expected calls."""
    warn_mock = MagicMock()
    stop_mock = MagicMock()
    monkeypatch.setattr(server, "_warn_default_token", warn_mock)
    monkeypatch.setattr(server._clicker, "stop", stop_mock)

    with TestClient(app) as client:
        # Startup hook should run _warn_default_token
        warn_mock.assert_called_once()
        # Shutdown hook should not have run yet
        stop_mock.assert_not_called()

    # After exit (shutdown hook), stop should run
    stop_mock.assert_called_once()


@pytest.mark.asyncio
async def test_monitor_no_duplicate_startup(monkeypatch):
    """Verify JarvisControllerV2.run_cli does not start the resource monitor a second time."""
    from core.controller_v2 import JarvisControllerV2

    mock_monitor = MagicMock()
    mock_monitor.start = AsyncMock()

    # Mock the input() function to raise EOFError so that run_cli exits immediately
    def mock_input(*args, **kwargs):
        raise EOFError()
    monkeypatch.setattr("core.controller_v2.input", mock_input, raising=False)

    # Setup controller with mocked monitor
    ctrl = JarvisControllerV2()
    ctrl.monitor = mock_monitor

    # Call start and run_cli (simulating entrypoint startup sequence)
    await ctrl.start()
    await ctrl.run_cli()

    # Verify start was scheduled exactly once (from ctrl.start())
    assert mock_monitor.start.call_count == 1
