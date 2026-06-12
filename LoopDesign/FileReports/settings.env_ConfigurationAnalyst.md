# File Report: settings.env
**Role:** Configuration Analyst

## File Overview
This file contains environment variables and secret keys used by Jarvis to integrate with external services and fallback mechanisms.

## Assumptions & Contracts
- **Format:** `.env` style key-value pairs.
- **Implicit Environment Assumptions:**
  - **Email Service:** Assumes a standard IMAP/SMTP setup, defaulting to Gmail servers (`smtp.gmail.com`, `imap.gmail.com`).
  - **Home Assistant:** Assumes a local mDNS-based Home Assistant instance at `http://homeassistant.local:8123`.
  - **Twilio Sandbox:** The WhatsApp integration (`whatsapp:+14155238886`) usually maps to Twilio's developer sandbox number, assuming the user is using Twilio for WhatsApp messaging.

## Secrets & Env Vars
The following variables are defined (many intentionally left blank for the user to fill):
- **Email:** `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`, `IMAP_HOST`
- **WhatsApp via Twilio:** `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM`
- **Home Assistant:** `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`
- **GitHub:** `GITHUB_TOKEN`, `GITHUB_DEFAULT_REPO`
- **Misc/Local:** `PORCUPINE_ACCESS_KEY`, `CALENDAR_ICS_PATH` (defaults to `data/calendar.ics`)
- **Cloud LLM Fallback:** `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `CLOUD_LLM_FALLBACK_ENABLED=true`

## Extracted Prompts
- None.

## Configuration Variables
All variables listed under "Secrets & Env Vars" function as the primary configuration overrides for the system.
