# Configuration Analyst Report: auth.db

## File Overview
- **Path**: `d:\AI\Jarvis\data\auth.db`
- **Type**: SQLite Database
- **Purpose**: Manages authentication and authorization credentials.

## Exhaustive Line-by-Line / Schema Analysis
The file is a binary SQLite database. Analysis of the schema reveals the following:

1. **Table: `users`**
   - `username TEXT PRIMARY KEY`: Expects a unique string identifier for users.
   - `password_hash TEXT NOT NULL`: Implicit assumption that passwords are hashed before storage. Data extraction revealed a bcrypt hash format (`bcrypt$$...`).
   - `is_admin INTEGER NOT NULL DEFAULT 1`: Represents boolean privileges. Defaults to admin.
   - `created_at REAL NOT NULL`: Stores UNIX timestamp (floating point).

2. **Table: `api_tokens`**
   - `token_hash TEXT PRIMARY KEY`: Implicitly assumes API tokens are hashed (not stored in plain text).
   - `label TEXT NOT NULL`: A descriptive name for the token.
   - `created_at REAL NOT NULL`: Token creation timestamp.
   - `last_used_at REAL`: Tracking usage metrics.

## Implicit Environment Assumptions
- **Authentication**: System inherently relies on local DB-based authentication rather than an external identity provider (OAuth/OIDC).
- **Crypto Dependency**: Relies on `bcrypt` (or similar library) for checking `password_hash` strings.
- **Admin Setup**: Contains a default user `admin_user`.

## Secrets & Env Vars
- No raw environment variables found.
- Secret observed: A hashed password for `admin_user`.
- API tokens table was empty, but expected to store token hashes.
