# `whatsapp.py` - API Analyst Report

## Overview
WhatsApp integration via the Twilio platform.

## Endpoints / Tools
1. `send_whatsapp`
   - Description: Send a WhatsApp message.
   - Risk: confirm (write)
   - Arguments: `to` (string, required, phone number with country code), `message` (string, required).

## External Contracts / Dependencies
- Requires `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM`.
- Uses the `twilio` python package (`twilio.rest.Client`).

## Assumptions
- Prepends `whatsapp:` to the `to` argument as required by the Twilio WhatsApp API.
- Re-initializes the `Client` on every request.
- Runs the blocking Twilio API client method inside `loop.run_in_executor`.
- Returns the Twilio message `sid` as data on success.
