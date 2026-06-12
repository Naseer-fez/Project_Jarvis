# API Analyst Report: security\auth.py

## Dependencies
- `from __future__ import annotations`
- `import contextlib`
- `import base64`
- `import hashlib`
- `import hmac`
- `import os`
- `import secrets`
- `import sqlite3`
- `import time`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Any`

## Configuration Variables
- `SESSION_COOKIE` = `'jarvis_session'`
- `CSRF_COOKIE` = `'jarvis_csrf'`
- `SESSION_TTL_S` = `60 * 60 * 12`
- `PBKDF2_ROUNDS` = `260000`

## Schemas & API Contracts (Classes)

### Class `AuthUser`
**Fields/Schema:**
  - `username: str`
  - `is_admin: bool`



### Class `AuthManager`
**Methods:**
- `def __init__(self, db_path: str | Path, secret_key: str | None=None) -> None`
- @contextlib.contextmanager
- `def _connect(self)`
- `def _init_db(self) -> None`
- `def bootstrap_admin_from_env(self) -> bool`
- `def user_count(self) -> int`
- `def create_user(self, username: str, password: str, *, is_admin: bool=True) -> None`
- `def authenticate(self, username: str, password: str) -> AuthUser | None`
- `def create_api_token(self, label: str='automation') -> str`
- `def verify_api_token(self, token: str) -> bool`
- `def sign_session(self, user: AuthUser) -> str`
- `def verify_session(self, token: str) -> AuthUser | None`
- `def make_csrf_token(self, session_token: str) -> str`
- `def verify_csrf_token(self, session_token: str, csrf_token: str) -> bool`
- `def hash_password(self, password: str) -> str`
- `def verify_password(self, password: str, password_hash: str) -> bool`
- `def _sign(self, payload: str) -> str`
- `def _token_hash(self, token: str) -> str`


## Functions & Endpoints

### `auth_db_from_config`
`def auth_db_from_config(config: Any) -> Path`