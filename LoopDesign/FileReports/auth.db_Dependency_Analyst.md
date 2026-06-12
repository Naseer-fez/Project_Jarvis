# Dependency Analysis: auth.db

## Overview
This file is an SQLite database used for authentication and authorization. 

## Schemas / API Contracts
The database assumes the following table schema:
- `users`:
  - `username` (TEXT PRIMARY KEY)
  - `password_hash` (TEXT NOT NULL)
  - `is_admin` (INTEGER NOT NULL DEFAULT 1)
  - `created_at` (REAL NOT NULL)
- `api_tokens`:
  - `token_hash` (TEXT PRIMARY KEY)
  - `label` (TEXT NOT NULL)
  - `created_at` (REAL NOT NULL)
  - `last_used_at` (REAL)

## Assumptions & Dependencies
- Depends on `sqlite3` driver.
- The system assumes role-based access control with `is_admin` acting as a boolean flag.
- Assumes timestamps are stored as `REAL` (floating-point numbers, likely UNIX epoch timestamps).
- API Tokens are hashed, assuming a one-way hashing mechanism is implemented in the caller code.
