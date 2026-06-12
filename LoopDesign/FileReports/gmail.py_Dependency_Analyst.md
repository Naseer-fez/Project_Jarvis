# File Report: gmail.py
## Role: Dependency Analyst

### 1. Library Requirements
- `aiohttp` (Third-party)
- `base64`, `os`, `email.mime.text`, `typing` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- Google OAuth API (`https://oauth2.googleapis.com/token`)
- Gmail API v1 (`https://gmail.googleapis.com/gmail/v1/users/me`)

### 3. Hidden Execution Links
- Asynchronously refreshes the access token using the refresh token on *every single* execution.
- `_summarize_unread` fetches unread emails and signals the router to use task_type="synthesis" for LLM summarization.
- `send_gmail` converts text body to base64 `urlsafe_b64encode` string for the raw payload.

### 4. Assumptions & API Contracts
- Email subjects are truncated to 200 chars. Snippets are strictly truncated to `_MAX_BODY_CHARS` (2000 chars) before returning them to context to prevent overwhelming the LLM or prompt injection via massive emails.
- `mark_as_read` explicitly removes the `"UNREAD"` label.
- Assumes the client has offline access to the user account (so a `refresh_token` exists).

### 5. Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`
- Internal: `_MAX_BODY_CHARS` (2000)

### 6. Prompts Found
- None.
