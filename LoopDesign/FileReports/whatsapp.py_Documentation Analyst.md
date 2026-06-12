# Documentation Report: clients/whatsapp.py

## Assumptions
- Uses Twilio as the backend service to send WhatsApp messages.
- Formats destination to `whatsapp:{to}` and maps sender to `TWILIO_WHATSAPP_FROM`.

## Schema / API Contract
- Tool: `send_whatsapp(to: str, message: str)`.

## Dependencies
- `twilio` (external)
- `os`, `asyncio` (stdlib)

## Configuration Variables
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_WHATSAPP_FROM`

## Prompts
None.
