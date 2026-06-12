# Documentation Report: clients/email.py

## Assumptions
- Uses standard Python `smtplib` and `imaplib` for SMTP and IMAP operations.
- Assumes `SMTP_PORT` defaults to 587 and supports STARTTLS.
- Searches IMAP by subject using `SUBJECT "query"`.
- Fetching retrieves latest messages and parses headers for From, Subject, Date.

## Schema / API Contract
- Tool: `send_email(to: str, subject: str, body: str)`
- Tool: `read_emails(folder: str, limit: int)`
- Tool: `search_emails(query: str)`

## Dependencies
- `smtplib`, `imaplib`, `email` (stdlib)

## Configuration Variables
- `EMAIL_ADDRESS`
- `EMAIL_PASSWORD`
- `SMTP_HOST`
- `SMTP_PORT` (optional, default 587)
- `IMAP_HOST`

## Prompts
None.
