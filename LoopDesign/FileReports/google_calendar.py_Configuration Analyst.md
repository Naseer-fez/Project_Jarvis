# Configuration Analysis: google_calendar.py
**Path**: d:\AI\Jarvis\integrations\clients\google_calendar.py

## 1. Environment & Configuration Variables
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET
- GOOGLE_REFRESH_TOKEN

## 2. Secrets & Credentials
Detected potential secret references: secret, token

## 3. Dependencies
- __future__
- aiohttp
- datetime
- integrations.base
- os
- typing

## 4. API Contracts & Tools (Schemas)
- Tool Schema: create_event
- Tool Schema: delete_event
- Tool Schema: find_free_slot
- Tool Schema: list_events

## 5. Implicit Assumptions (URLs, hardcoded paths)
### URLs
- https://oauth2.googleapis.com/token
- https://www.googleapis.com/calendar/v3

## 6. Prompts
None detected.
