# Security Specification

## 1. Security Architecture Overview
The security model of Jarvis relies on a centralized `AuthManager` with a lightweight SQLite backend (`auth.db`) handling identity and sessions, paired with a dynamic `PermissionMatrix` governing action execution. The dashboard and API endpoints are protected using FastAPI routing hooks, with both session cookies and token-based authentication supported.

## 2. Authentication Flows
- **Human Authentication**:
  - `POST` to `/login` with `username` and `password`.
  - Passwords are cryptographically hashed using `bcrypt` (12 rounds) if the dependency is installed, or fallback to PBKDF2-HMAC-SHA256 (260,000 rounds) with a 16-byte random salt.
  - On success, issues a signed `jarvis_session` cookie valid for 12 hours.
  - Cookie attributes: `HttpOnly=True`, `SameSite=lax`, `Secure=False`. Note: `Secure=False` is hardcoded, meaning it relies on a reverse proxy for HTTPS termination without natively enforcing secure cookies.
- **API/Automation Authentication**:
  - Handled via the `X-Dashboard-Token` header.
  - The system checks if it matches an active API token stored in the database (hashed validation).
  - Fallback: Checks against the static `JARVIS_DASHBOARD_TOKEN` environment variable.
- **WebSocket Authentication**:
  - Checked via `token` query parameter or `jarvis_session` cookie, allowing seamless real-time communication for authorized sessions.

## 3. Session Management
- Sessions are stateless and cryptographically signed (no server-side session store).
- Payload format: `username|is_admin|expires|nonce|signature`.
- Signature is generated using HMAC-SHA256 using the master `JARVIS_SECRET_KEY`.
- Expiration (`SESSION_TTL_S`) is 12 hours.
- **CSRF Protection**: Nonce-based CSRF tokens (`make_csrf_token`) signed with HMAC-SHA256. The CSRF tokens are mathematically tied to the active session token to prevent cross-session replay attacks.

## 4. Authorization & Roles
- **Roles**: The system defines a simplified RBAC model via the `AuthUser` data class, primarily distinguishing `is_admin` flags. By default, newly created users and bootstrapped users from environment variables have `is_admin=True`.
- **Enforcement**: Access to dashboard pages and API endpoints (e.g., `/api/clicker/*`, `/command`) is gated by the `_is_authorized` check in the FastAPI application. Unauthorized access yields a 303 Redirect to `/login` for GUI routes or a 401 JSON response for APIs.

## 5. Permissions & Access Rules
Governed by the `PermissionMatrix` (`core/permission_matrix.py`) which evaluates commands against risk tiers defined in `jarvis.ini`.
- **Blocked/Forbidden Actions**: Actions matching `blocked_actions`, `critical_actions`, or `forbidden_actions` (e.g., `format_disk`, `wipe_disk`, `registry_write`, `shell_exec`, `file_delete`) are outright denied and return a `blocked_actions` array.
- **Confirmation Required**: Actions matching `user_confirmed_actions` or `high_risk_actions` (e.g., `write_file`, `process_spawn`, `click_screen_target`, `type_text`, `launch_application`) yield a `needs_confirmation` state. These operations halt autonomous execution until human approval is provided.
- **Allowed Auto-execution**: `medium_risk_actions` (e.g., `read_file`, `web_search`) and `low_risk_actions` (e.g., `status`, `health_check`) execute transparently without interruption.
- **Execution Failsafes**: `failsafe_auto_disable_on_error = true` with a threshold of 3 errors prevents runaway loops and automation mistakes.

## 6. Secrets & Encryption
- **Cryptography**: Uses standard library `hashlib`, `hmac`, and `secrets`.
- **Secret Key**: `JARVIS_SECRET_KEY` is loaded from the environment or config. If missing, it falls back to `"development-only-secret"` with a startup warning logged via `logging.warning`.
- **Token Hashing**: API tokens are generated using 32 bytes of url-safe entropy. The plaintext token is returned to the user only once upon creation; the database only stores the HMAC-SHA256 digest of the token to prevent leakage if the database is compromised.
- **Encryption at Rest**: There is no encryption at rest. The `auth.db` and memory databases (SQLite) are stored in plaintext on disk. Environment variables and `.ini` files are used for sensitive external API keys (e.g., `tavily_api_key`).
