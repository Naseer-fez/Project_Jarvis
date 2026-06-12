# File Report: config/settings.env

## Purpose
This file holds sensitive environment variables and secrets needed by various integrations of Jarvis.

## Responsibilities
- Secure storage of credentials.
- Configuration of third-party APIs such as Email (SMTP/IMAP), Twilio (WhatsApp), Home Assistant, GitHub, Porcupine (Wake word), and cloud LLMs (Groq, OpenAI, Anthropic).

## Architecture Role
Serves as the `.env` file that is typically loaded on application startup to populate `os.environ`. These variables are used by the respective integration tools (e.g., Email tool, Home Assistant tool, Cloud LLM fallback providers).

## Content Breakdown
- **Email**: SMTP/IMAP settings.
- **WhatsApp**: Twilio SID, Token, and sender number.
- **Home Assistant**: URL and Token.
- **GitHub**: Token and Default Repo.
- **Wake Word**: Porcupine Access Key.
- **Cloud LLM**: Groq, OpenAI, Anthropic keys, and a toggle `CLOUD_LLM_FALLBACK_ENABLED`.
- **Other**: `CALENDAR_ICS_PATH=data/calendar.ics`.

## Prompts
- None.
