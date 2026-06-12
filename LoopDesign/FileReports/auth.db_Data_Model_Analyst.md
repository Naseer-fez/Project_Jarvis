# Data Model Analyst Report: auth.db

## File Analysis
- **Filename**: `auth.db`
- **Path**: `d:\AI\Jarvis\data\auth.db`
- **Format**: SQLite3 Database

## Schema and State Objects
The file contains two main tables related to user authentication and API token management:

### Table: `users`
```sql
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
```
- **username**: Primary identifier.
- **password_hash**: Stored hash for the user password (likely bcrypt/argon2 or similar).
- **is_admin**: Integer acting as a boolean flag. Defaults to 1 (all created users are admins by default).
- **created_at**: Real type storing a unix timestamp or floating point epoch.

### Table: `api_tokens`
```sql
CREATE TABLE api_tokens (
    token_hash TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used_at REAL
);
```
- **token_hash**: Hashed version of the token to prevent raw leak if DB is compromised.
- **label**: Human-readable name for the token.
- **created_at**: Real type for token creation timestamp.
- **last_used_at**: Real type for tracking token utilization timestamp.

## Assumptions & Contracts
- The system assumes the presence of at least one user, defaulting to admin privileges.
- API tokens are tracked for usage and hashed for security.
- Timestamps are stored as REAL, which usually implies Python's `time.time()` output (Unix epoch seconds).

## Dependencies & Variables
- Relied upon by the authentication and authorization middleware of the API/System.

## Extracted Prompts
None found.
