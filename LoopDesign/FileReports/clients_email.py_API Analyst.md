# `email.py` - API Analyst Report

## Overview
Email integration using the standard library `smtplib` and `imaplib` for sending and reading emails via SMTP and IMAP.

## Endpoints / Tools
1. `send_email`
   - Description: Send an email.
   - Risk: confirm (write)
   - Arguments: `to` (string, required), `subject` (string, required), `body` (string, required).
2. `read_emails`
   - Description: Read recent emails from inbox.
   - Risk: low (read-only)
   - Arguments: `folder` (string, default "INBOX"), `limit` (integer, default 10).
3. `search_emails`
   - Description: Search emails by keyword (subject).
   - Risk: low (read-only)
   - Arguments: `query` (string, required).

## External Contracts / Dependencies
- Relies on `smtplib` and `imaplib` (standard Python libraries).
- Requires environment variables: `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `SMTP_HOST`, `IMAP_HOST`. Optionally uses `SMTP_PORT` (default 587).

## Assumptions
- SMTP server uses TLS via `starttls()` and the port defaults to 587.
- IMAP server uses SSL via `IMAP4_SSL`.
- In `read_emails`, emails are fetched sequentially and only headers "From", "Subject", "Date" are extracted.
- In `search_emails`, the search is scoped strictly to `SUBJECT "{safe_query}"` and quotes are stripped to prevent IMAP injection-like errors.
- Network operations block the thread, so they are wrapped in `loop.run_in_executor`.
