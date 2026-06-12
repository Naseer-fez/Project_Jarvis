# API Analyst Report: auth.db

## Overview
This SQLite database manages authentication within the Jarvis system.

## Schema / Structure
```sql
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE TABLE api_tokens (
    token_hash TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used_at REAL
);
```

## API Contracts & Data Schema
- **`users` Table Contract**:
  - `username` is the primary identifier.
  - Authentication must verify against `password_hash`.
  - Roles are inferred via `is_admin` boolean (default 1), meaning new users may default to admin access unless explicitly configured otherwise.
  - `created_at` stored as a `REAL` (likely an epoch timestamp).
- **`api_tokens` Table Contract**:
  - API consumers authenticate with a hashed token (`token_hash`), matched against an incoming raw token.
  - `label` allows identifying the token's purpose or owner.
  - `created_at` and `last_used_at` tracked as `REAL` (epoch timestamps).

## Assumptions
- Jarvis exposes APIs that require token-based authentication.
- Any new API endpoints added to the system will likely need to inject an authentication middleware that checks the `api_tokens` table.
- Passwords are exchanged securely and hashed prior to storage.
