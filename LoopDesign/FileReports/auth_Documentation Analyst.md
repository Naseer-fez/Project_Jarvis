# Analysis Report for auth.py

## Dependencies
- __future__.annotations
- contextlib
- base64
- hashlib
- hmac
- os
- secrets
- sqlite3
- time
- dataclasses.dataclass
- pathlib.Path
- typing.Any

## Schemas
- AuthUser
- AuthUser attribute: username
- AuthUser attribute: is_admin
- AuthManager

## API Contracts
- AuthManager.__init__(self, db_path, secret_key)
- AuthManager._connect(self)
- AuthManager._init_db(self)
- AuthManager.bootstrap_admin_from_env(self)
- AuthManager.user_count(self)
- AuthManager.create_user(self, username, password)
- AuthManager.authenticate(self, username, password)
- AuthManager.create_api_token(self, label)
- AuthManager.verify_api_token(self, token)
- AuthManager.sign_session(self, user)
- AuthManager.verify_session(self, token)
- AuthManager.make_csrf_token(self, session_token)
- AuthManager.verify_csrf_token(self, session_token, csrf_token)
- AuthManager.hash_password(self, password)
- AuthManager.verify_password(self, password, password_hash)
- AuthManager._sign(self, payload)
- AuthManager._token_hash(self, token)
- auth_db_from_config(config)

## Configuration Variables
- SESSION_COOKIE
- CSRF_COOKIE
- SESSION_TTL_S
- PBKDF2_ROUNDS

## Assumptions & Notes
None

