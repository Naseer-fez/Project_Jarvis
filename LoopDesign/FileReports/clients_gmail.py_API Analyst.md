# `gmail.py` - API Analyst Report

## Overview
Gmail integration using the async `aiohttp` library to interact with Gmail API v1.

## Endpoints / Tools
1. `list_unread`
   - Description: List unread emails from Gmail inbox.
   - Risk: low (read-only)
   - Arguments: `max_results` (integer, default 10).
2. `send_gmail`
   - Description: Send an email via Gmail.
   - Risk: confirm (write)
   - Arguments: `to` (string, required), `subject` (string, required), `body` (string, plain-text).
3. `summarize_unread`
   - Description: Fetch unread emails and return truncated content for LLM summarization.
   - Risk: low (read-only)
   - Arguments: `max_results` (integer, default 5).
4. `mark_as_read`
   - Description: Mark a Gmail message as read by its message ID.
   - Risk: confirm (write)
   - Arguments: `message_id` (string, required).

## External Contracts / Dependencies
- Requires `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`.
- Interacts with `https://oauth2.googleapis.com/token` to fetch the access token via the refresh token.
- Uses `https://gmail.googleapis.com/gmail/v1/users/me` base API URL.
- Depends on `aiohttp` for async HTTP requests.

## Assumptions
- Always truncates email content/snippets to 2000 characters before returning to avoid polluting context window or LLM injections.
- `summarize_unread` uses `task_type="synthesis"`.
- Requests `format=metadata` with headers `["From", "Subject", "Date"]` to minimize payload sizes during `list_unread`.
- `send_gmail` uses `base64.urlsafe_b64encode` to encode plain-text `MIMEText` objects for the `raw` payload field.
- Maximum results for `list_unread` is clamped at 50 to avoid hitting API rate limits or excessive loading.
