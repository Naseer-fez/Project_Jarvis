# Configuration Analysis: email.py
**Path**: d:\AI\Jarvis\integrations\clients\email.py

## 1. Environment & Configuration Variables
- EMAIL_ADDRESS
- EMAIL_PASSWORD
- IMAP_HOST
- SMTP_HOST
- SMTP_PORT

## 2. Secrets & Credentials
Detected potential secret references: password

## 3. Dependencies
- __future__
- asyncio
- email
- email.mime.text
- imaplib
- integrations.base
- os
- smtplib
- typing

## 4. API Contracts & Tools (Schemas)
- Tool Schema: read_emails
- Tool Schema: search_emails
- Tool Schema: send_email

## 5. Implicit Assumptions (URLs, hardcoded paths)
No hardcoded URLs detected.

## 6. Prompts
None detected.
