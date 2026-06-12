# clients/email.py API Analyst Report

## Overview
Email integration providing basic SMTP sending and IMAP reading/searching functionality.

## API Contracts & Methods
- `EmailIntegration(BaseIntegration)`
  - `is_available()`: Checks for stdlib `smtplib`/`imaplib` and required config variables.

## Tools Exposed
- `send_email`
  - **Risk:** `confirm`
  - **Args:** `to` (str), `subject` (str), `body` (str)
  - **Behavior:** Logs in via `smtplib.SMTP(host, port)` and `starttls()`.
- `read_emails`
  - **Risk:** `low`
  - **Args:** `folder` (str, default "INBOX"), `limit` (int, default 10)
  - **Behavior:** Fetches recent emails via `IMAP4_SSL`.
- `search_emails`
  - **Risk:** `low`
  - **Args:** `query` (str)
  - **Behavior:** IMAP search using `SUBJECT "{query}"`.

## Configuration Variables
- `EMAIL_ADDRESS`
- `EMAIL_PASSWORD`
- `SMTP_HOST`
- `IMAP_HOST`
- `SMTP_PORT` (defaults to "587" if not set)

## Assumptions & Constants
- Timeout for network operations is set to 10 seconds.
- Sent emails are MIMEText plain text format.
- Only retrieves `"From"`, `"Subject"`, and `"Date"` fields when reading emails (no body parsing).

## Dependencies
- Standard library: `smtplib`, `imaplib`, `email`, `asyncio`, `os`.

## Prompts
- None.
