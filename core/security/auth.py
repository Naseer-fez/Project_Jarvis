from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import bcrypt  # type: ignore[import]
except Exception:  # pragma: no cover - optional optimized hasher
    bcrypt = None  # type: ignore[assignment]


SESSION_COOKIE = "jarvis_session"
CSRF_COOKIE = "jarvis_csrf"
SESSION_TTL_S = 60 * 60 * 12
PBKDF2_ROUNDS = 260_000


@dataclass(frozen=True)
class AuthUser:
    username: str
    is_admin: bool = True


class AuthManager:
    def __init__(self, db_path: str | Path, secret_key: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_key = secret_key or os.environ.get("JARVIS_SECRET_KEY", "")
        if not self.secret_key:
            self.secret_key = "development-only-secret"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_tokens (
                    token_hash TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL
                )
                """
            )

    def bootstrap_admin_from_env(self) -> bool:
        username = os.environ.get("JARVIS_ADMIN_USER", "").strip()
        password = os.environ.get("JARVIS_ADMIN_PASSWORD", "")
        if not username or not password:
            return False
        if self.user_count() > 0:
            return False
        self.create_user(username, password, is_admin=True)
        return True

    def user_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password: str, *, is_admin: bool = True) -> None:
        username = username.strip()
        if not username:
            raise ValueError("username is required")
        if len(password) < 12:
            raise ValueError("password must be at least 12 characters")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users(username, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, self.hash_password(password), int(is_admin), time.time()),
            )

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, is_admin FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not self.verify_password(password, str(row["password_hash"])):
            return None
        return AuthUser(username=str(row["username"]), is_admin=bool(row["is_admin"]))

    def create_api_token(self, label: str = "automation") -> str:
        token = secrets.token_urlsafe(32)
        digest = self._token_hash(token)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_tokens(token_hash, label, created_at) VALUES (?, ?, ?)",
                (digest, label, time.time()),
            )
        return token

    def verify_api_token(self, token: str) -> bool:
        if not token:
            return False
        digest = self._token_hash(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token_hash FROM api_tokens WHERE token_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE api_tokens SET last_used_at = ? WHERE token_hash = ?",
                (time.time(), digest),
            )
        return True

    def sign_session(self, user: AuthUser) -> str:
        expires = int(time.time() + SESSION_TTL_S)
        nonce = secrets.token_urlsafe(12)
        payload = f"{user.username}|{int(user.is_admin)}|{expires}|{nonce}"
        sig = self._sign(payload)
        raw = f"{payload}|{sig}".encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def verify_session(self, token: str) -> AuthUser | None:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            username, is_admin, expires, nonce, sig = raw.split("|", 4)
        except Exception:
            return None
        del nonce
        payload = "|".join([username, is_admin, expires, raw.split("|", 4)[3]])
        if not hmac.compare_digest(sig, self._sign(payload)):
            return None
        if int(expires) < int(time.time()):
            return None
        return AuthUser(username=username, is_admin=bool(int(is_admin)))

    def make_csrf_token(self, session_token: str) -> str:
        nonce = secrets.token_urlsafe(18)
        sig = self._sign(f"{session_token}|{nonce}")
        return f"{nonce}.{sig}"

    def verify_csrf_token(self, session_token: str, csrf_token: str) -> bool:
        if "." not in csrf_token:
            return False
        nonce, sig = csrf_token.split(".", 1)
        expected = self._sign(f"{session_token}|{nonce}")
        return hmac.compare_digest(sig, expected)

    def hash_password(self, password: str) -> str:
        if bcrypt is not None:
            hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
            return "bcrypt$" + hashed.decode("ascii")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
        return "pbkdf2_sha256${}${}${}".format(
            PBKDF2_ROUNDS,
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )

    def verify_password(self, password: str, password_hash: str) -> bool:
        if password_hash.startswith("bcrypt$") and bcrypt is not None:
            return bool(
                bcrypt.checkpw(
                    password.encode("utf-8"),
                    password_hash.removeprefix("bcrypt$").encode("ascii"),
                )
            )
        if password_hash.startswith("pbkdf2_sha256$"):
            _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(digest_b64.encode("ascii"))
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                int(rounds),
            )
            return hmac.compare_digest(actual, expected)
        return False

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self.secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _token_hash(self, token: str) -> str:
        return hmac.new(
            self.secret_key.encode("utf-8"),
            token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()


def auth_db_from_config(config: Any) -> Path:
    try:
        raw = config.get("security", "auth_db", fallback="data/auth.db")
    except Exception:
        raw = "data/auth.db"
    path = Path(raw)
    if path.is_absolute():
        return path
    from core.runtime.bootstrap import PROJECT_ROOT

    return PROJECT_ROOT / path


__all__ = ["AuthManager", "AuthUser", "SESSION_COOKIE", "CSRF_COOKIE", "auth_db_from_config"]
