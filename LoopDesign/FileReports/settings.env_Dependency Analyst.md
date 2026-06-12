# Dependency Analysis: settings.env

## 1. Schema / API Contract
- Format: Standard `.env` format (Key=Value), parsed by `dotenv` or equivalent.
- Handles secrets, overriding and extending configurations found in `jarvis.ini`.

## 2. Library Requirements / Service Dependencies
- **Email Services**: SMTP (`smtp.gmail.com:587`) and IMAP (`imap.gmail.com`).
- **Twilio**: External provider API needed for WhatsApp messaging.
- **Home Assistant**: Local network integration (`http://homeassistant.local:8123`).
- **GitHub**: Requires a Personal Access Token (`GITHUB_TOKEN`) for repo manipulation.
- **Picovoice Porcupine**: Wakeword engine (implied by `PORCUPINE_ACCESS_KEY`).
- **Cloud LLMs**: `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Used when local models fail or for fallback requests.

## 3. Configuration Variables
- Boolean control over fallback: `CLOUD_LLM_FALLBACK_ENABLED=true`.
- Hardcoded reference to calendar export: `CALENDAR_ICS_PATH=data/calendar.ics`.
- Explicit Twilio WhatsApp from-number (`whatsapp:+14155238886`).

## 4. Hidden Execution Links
- Binds external SaaS/cloud APIs into the core framework.
- If cloud APIs fail, behavior depends entirely on logic outside this file, though the `FALLBACK_ENABLED=true` indicates the system anticipates failovers.
- `PORCUPINE_ACCESS_KEY` must correlate to the `wakeword_model` parameter in `jarvis.ini` (which says `hey_jarvis`), implying the system switches between local and cloud wakewords or Porcupine is the backend for `hey_jarvis`.
