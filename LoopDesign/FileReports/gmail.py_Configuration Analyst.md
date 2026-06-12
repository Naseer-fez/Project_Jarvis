# Configuration Analysis: gmail.py
**Path**: d:\AI\Jarvis\integrations\clients\gmail.py

## 1. Environment & Configuration Variables
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET
- GOOGLE_REFRESH_TOKEN

## 2. Secrets & Credentials
Detected potential secret references: secret, token

## 3. Dependencies
- __future__
- aiohttp
- base64
- email.mime.text
- integrations.base
- os
- typing

## 4. API Contracts & Tools (Schemas)
- Tool Schema: list_unread
- Tool Schema: mark_as_read
- Tool Schema: send_gmail
- Tool Schema: summarize_unread

## 5. Implicit Assumptions (URLs, hardcoded paths)
### URLs
- https://gmail.googleapis.com/gmail/v1/users/me
- https://oauth2.googleapis.com/token

## 6. Prompts
None detected.
