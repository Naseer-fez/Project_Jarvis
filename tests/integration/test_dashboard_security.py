import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from dashboard.server import app, set_controller
import dashboard.server as server
from core.security.auth import AuthManager


@pytest.fixture
def isolated_project_root(tmp_path, monkeypatch):
    """Provides an isolated PROJECT_ROOT to prevent global mutation."""
    project_root = tmp_path / "jarvis_project"
    project_root.mkdir()
    monkeypatch.setattr(server, "PROJECT_ROOT", project_root)
    return project_root


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


def test_unauthorized_view_file_redirects(monkeypatch):
    """Verify unauthorized calls to /api/view-file redirect to login page."""
    monkeypatch.setattr(server, "_auth_manager", None)
    client = TestClient(app)
    response = client.get("/api/view-file?path=config/jarvis.ini", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_authorized_view_file_success(test_auth_db, mock_dashboard_controller, isolated_project_root, monkeypatch):
    """Verify authorized path viewing works for files inside project root."""
    auth, _ = test_auth_db
    monkeypatch.setattr(server, "_auth_manager", auth)
    set_controller(mock_dashboard_controller)

    client = TestClient(app)
    user = auth.authenticate("admin_user", "securepassword123")
    valid_session = auth.sign_session(user)

    client.cookies.set("jarvis_session", valid_session)

    # Let's create a test file inside the isolated project root
    test_file_path = isolated_project_root / "test_temp_doc.txt"
    test_file_path.write_text("Hello Jarvis secure area", encoding="utf-8")

    response = client.get(f"/api/view-file?path={test_file_path}")
    assert response.status_code == 200
    assert response.text == "Hello Jarvis secure area"


def test_authorized_view_file_traversal_blocked(test_auth_db, mock_dashboard_controller, isolated_project_root, tmp_path, monkeypatch):
    """Verify authorized path viewing blocks path traversal outside PROJECT_ROOT."""
    auth, _ = test_auth_db
    monkeypatch.setattr(server, "_auth_manager", auth)
    set_controller(mock_dashboard_controller)

    client = TestClient(app)
    user = auth.authenticate("admin_user", "securepassword123")
    valid_session = auth.sign_session(user)

    client.cookies.set("jarvis_session", valid_session)

    # File outside isolated PROJECT_ROOT (in the tmp_path parent)
    outside_file = tmp_path / "outside_secret.txt"
    outside_file.write_text("classified document", encoding="utf-8")

    # Path traversal with relative parent dirs or absolute outside paths
    paths_to_test = [
        str(outside_file),
        f"../../../../../../{outside_file.name}",
        f"..\\..\\..\\..\\..\\..\\{outside_file.name}",
        f"outputs/../../{outside_file.name}",
        f"outputs/..\\..\\{outside_file.name}",
        "/etc/passwd",
        "C:\\windows\\win.ini",
        "C:/windows/win.ini",
        "\\\\.\\C:\\",
        "../../../../../../etc/passwd",
        # Null byte injection
        f"{outside_file}%00",
        f"../../../../../../{outside_file.name}%00",
        # URL encoded paths (using %2e for dot and %2f/%5c for slash/backslash)
        f"..%2f..%2f..%2f..%2f..%2f..%2f{outside_file.name}",
        f"..%5c..%5c..%5c..%5c..%5c..%5c{outside_file.name}",
        f"%2e%2e%2f%2e%2e%2f%2e%2e%2f{outside_file.name}",
    ]

    for test_path in paths_to_test:
        response = client.get(f"/api/view-file?path={test_path}")
        assert response.status_code == 403
        assert "Access denied" in response.json().get("error", "")


def test_dashboard_lifespan(monkeypatch):
    """Verify that dashboard lifespan startup and shutdown hooks trigger expected calls."""
    warn_mock = MagicMock()
    monkeypatch.setattr(server, "_warn_default_token", warn_mock)

    with TestClient(app):
        # Startup hook should run _warn_default_token
        warn_mock.assert_called_once()


@pytest.mark.asyncio
async def test_monitor_no_duplicate_startup(monkeypatch):
    """Verify JarvisControllerV2.run_cli does not start the resource monitor a second time."""
    from unittest.mock import AsyncMock
    from core.controller_v2 import JarvisControllerV2

    mock_monitor = MagicMock()
    
    # Use AsyncMock for coroutine methods so they can be awaited
    mock_monitor.start = AsyncMock()
    mock_monitor.stop = AsyncMock()

    import asyncio
    real_loop = asyncio.get_running_loop()

    def mock_run_in_executor(executor, func, *args, **kwargs):
        fut = real_loop.create_future()
        if func == input or getattr(func, "__name__", "") == "input":
            fut.set_exception(EOFError())
        else:
            fut.set_result(None)
        return fut
    
    mock_loop = MagicMock()
    mock_loop.run_in_executor = mock_run_in_executor
    monkeypatch.setattr("asyncio.get_running_loop", lambda: mock_loop, raising=False)

    # Setup controller with mocked monitor
    ctrl = JarvisControllerV2()
    ctrl.monitor = mock_monitor

    # Call start and run_cli (simulating entrypoint startup sequence)
    await ctrl.start()
    try:
        await ctrl.run_cli()
    except EOFError:
        pass # Handle EOFError naturally if raised
    finally:
        await ctrl.shutdown()

    # Verify start was scheduled exactly once (from ctrl.start())
    assert mock_monitor.start.call_count == 1


def test_authorized_view_file_restricted_roots(test_auth_db, mock_dashboard_controller, isolated_project_root, monkeypatch):
    """Verify that viewing sensitive file types or non-approved directories is blocked."""
    auth, _ = test_auth_db
    monkeypatch.setattr(server, "_auth_manager", auth)
    set_controller(mock_dashboard_controller)

    client = TestClient(app)
    user = auth.authenticate("admin_user", "securepassword123")
    valid_session = auth.sign_session(user)

    client.cookies.set("jarvis_session", valid_session)

    # 1. Accessing a file inside an allowed subdirectory (e.g. outputs) should be allowed
    outputs_dir = isolated_project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    test_out_file = outputs_dir / "test_run_output.txt"
    test_out_file.write_text("allowed outputs data", encoding="utf-8")

    response = client.get(f"/api/view-file?path={test_out_file}")
    assert response.status_code == 200
    assert response.text == "allowed outputs data"

    # 2. Accessing a sensitive file in the root directory (like .env) should be blocked (403)
    response = client.get("/api/view-file?path=.env")
    assert response.status_code == 403
    assert "Access denied" in response.json().get("error", "")

    # 3. Accessing a file in a non-approved subdirectory (like config/jarvis.ini) should be blocked (403)
    response = client.get("/api/view-file?path=config/jarvis.ini")
    assert response.status_code == 403
    assert "Access denied" in response.json().get("error", "")


def test_memory_page_unauthorized():
    client = TestClient(app)
    response = client.get("/memory", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"



