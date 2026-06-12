# File Report: settings.env
**Role**: Documentation Analyst

## 1. Assumptions
- This file holds secret keys and credentials necessary for external integrations.
- It acts as an environment variable definition file (`.env`).
- Missing values (e.g. `EMAIL_ADDRESS=`) are meant to be populated by the user.

## 2. Schema
- Key-Value pairs separated by `=`.
- Organized with comments grouping related keys (Email, WhatsApp, Home Assistant, GitHub, Cloud LLM Fallback).

## 3. API Contracts
- **Email**: SMTP (`smtp.gmail.com:587`) and IMAP (`imap.gmail.com`).
- **Twilio**: WhatsApp API integration (from `whatsapp:+14155238886`).
- **Home Assistant**: REST API (`http://homeassistant.local:8123`).
- **GitHub**: API integration.
- **Picovoice Porcupine**: Wakeword engine.
- **Cloud LLMs**: Groq, OpenAI, Anthropic APIs.

## 4. Dependencies
- Dependent on internet connectivity to reach cloud services.
- Local dependency on an `.ics` file path for calendars (`data/calendar.ics`).

## 5. Configuration Variables
- `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`, `IMAP_HOST`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM`
- `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`
- `GITHUB_TOKEN`, `GITHUB_DEFAULT_REPO`
- `PORCUPINE_ACCESS_KEY`
- `CALENDAR_ICS_PATH`
- `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `CLOUD_LLM_FALLBACK_ENABLED=true`

## 6. Prompts
- None found.
