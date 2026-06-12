# clients/gmail.py API Analyst Report

## Overview
Gmail API v1 integration for reading, sending, and summarizing emails via async HTTP requests using OAuth2.

## API Contracts & Methods
- `GmailIntegration(BaseIntegration)`
  - Uses `aiohttp` for REST calls to `https://gmail.googleapis.com/gmail/v1/users/me`.
  - Retrieves OAuth token directly using refresh token endpoint.

## Tools Exposed
- `list_unread(max_results=10)` [Risk: `low`]
- `send_gmail(to, subject, body)` [Risk: `confirm`]
  - Constructs `MIMEText` and base64 encodes it before POSTing.
- `summarize_unread(max_results=5)` [Risk: `low`]
  - Returns explicit `task_type="synthesis"` for router targeting.
- `mark_as_read(message_id)` [Risk: `confirm`]
  - Removes `UNREAD` label via modify endpoint.

## Assumptions & Constants
- `_MAX_BODY_CHARS = 2000`: Hardcoded truncation limit for email bodies to prevent prompt injection and context overflow.
- No raw headers injected blindly into context.

## Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

## Dependencies
- `aiohttp`
- `base64`, `email.mime.text.MIMEText`

## Prompts
- None.
