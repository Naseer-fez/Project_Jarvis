import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from dashboard.server import app, set_controller, _ws_is_authorized
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


def test_dashboard_routes_redirect_when_unauthorized():
    # Ensure no active auth manager or cookies
    server._auth_manager = None
    client = TestClient(app)
    
    # Page routes redirect to /login
    for path in ["/", "/memory", "/goals"]:
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_dashboard_login_page_renders():
    client = TestClient(app)
    response = client.get("/login")
    assert response.status_code == 200
    assert "JARVIS SYSTEM ACCESS" in response.text


def test_dashboard_login_success_and_logout(test_auth_db, mock_dashboard_controller):
    auth, db_file = test_auth_db
    server._auth_manager = auth
    set_controller(mock_dashboard_controller)
    
    client = TestClient(app)
    
    # 1. Login with invalid password
    response = client.post("/login", data={"username": "admin_user", "password": "wrongpassword"})
    assert response.status_code == 200
    assert "Invalid username or password" in response.text
    
    # 2. Login with valid password
    response = client.post("/login", data={"username": "admin_user", "password": "securepassword123"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "jarvis_session" in response.cookies
    
    # Get session cookie
    session_cookie = response.cookies["jarvis_session"]
    
    # 3. Access protected route with session cookie
    response = client.get("/", cookies={"jarvis_session": session_cookie})
    assert response.status_code == 200
    assert "Jarvis Dashboard" in response.text
    
    # 4. Command post with session cookie
    response = client.post("/command", json={"text": "hello jarvis"}, cookies={"jarvis_session": session_cookie})
    assert response.status_code == 200
    assert response.json() == {"response": "processed ok"}
    
    # 5. Logout deletes cookie
    response = client.get("/logout", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    
    # Verify cookie is cleared or deleted (empty or max-age=0/expires in past)
    # Different TestClient versions handle deleted cookies differently, but accessing protected page without cookie must redirect
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303


def test_ws_authorization(test_auth_db):
    auth, db_file = test_auth_db
    server._auth_manager = auth
    
    # Create mock WebSocket
    class MockWebSocket:
        def __init__(self, cookies):
            self.cookies = cookies
            
    # WebSocket auth succeeds with valid session cookie
    user = auth.authenticate("admin_user", "securepassword123")
    valid_session = auth.sign_session(user)
    
    ws_mock = MockWebSocket(cookies={"jarvis_session": valid_session})
    assert _ws_is_authorized(ws_mock, token="") is True
    
    # WebSocket auth succeeds with valid token query parameter
    ws_mock_empty = MockWebSocket(cookies={})
    assert _ws_is_authorized(ws_mock_empty, token=valid_session) is True
    
    # WebSocket auth fails with invalid/missing token
    assert _ws_is_authorized(ws_mock_empty, token="") is False
    assert _ws_is_authorized(ws_mock_empty, token="invalidtoken") is False
