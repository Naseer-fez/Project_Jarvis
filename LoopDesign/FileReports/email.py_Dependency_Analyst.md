# File Report: email.py
## Role: Dependency Analyst

### 1. Library Requirements
- `asyncio`, `email`, `imaplib`, `os`, `smtplib`, `typing` (Standard Library)
- `email.mime.text` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- Any valid SMTP and IMAP service (e.g. Gmail, Outlook, private server).
- Relies on external ports (587 default for SMTP, implicitly standard SSL port for IMAP).

### 3. Hidden Execution Links
- Uses standard library `smtplib.SMTP(..., timeout=10)` with `starttls()` and `login()`.
- Uses `imaplib.IMAP4_SSL` for searching/fetching emails.
- IMAP fetch uses `(RFC822)` standard to fetch raw bytes, then relies on `email_lib.message_from_bytes`.

### 4. Assumptions & API Contracts
- Expects plain text body payloads (MIMEText).
- SMTP Port defaults to `587` if not provided.
- `search_emails` uses simple IMAP `SUBJECT` search string logic and limits results to 10.
- `read_emails` pulls IDs and processes the last N items (reversed).

### 5. Configuration Variables
- `EMAIL_ADDRESS`
- `EMAIL_PASSWORD`
- `SMTP_HOST`
- `IMAP_HOST`
- Optional: `SMTP_PORT` (defaults to "587")

### 6. Prompts Found
- None.
