# Documentation Report: clients/gmail.py

## Assumptions
- Uses Gmail API v1 through raw REST calls (`aiohttp`), rather than google-api-python-client.
- Relies on offline `refresh_token` flow to get short-lived `access_token` automatically.
- Email contents are truncated to `2000` chars explicitly to avoid token explosion.
- Unread summary tool uses `task_type="synthesis"` which is likely intercepted by the LLM routing layer.

## Schema / API Contract
- Tools: `list_unread`, `send_gmail`, `summarize_unread`, `mark_as_read`.
- `send_gmail` expects a plain-text email body and builds a base64 encoded MIME object.

## Dependencies
- `aiohttp` (external)
- `base64`, `os`, `email.mime.text` (stdlib)

## Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

## Prompts
None.
