# File Report: email.py
**Path**: `d:\AI\Jarvis\integrations\clients\email.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- asyncio
- email
- imaplib
- os
- smtplib
- email.mime.text.MIMEText
- typing.Any
- integrations.base.BaseIntegration
- smtplib
- imaplib

## Classes and State Objects
### `EmailIntegration`
**Variables**: name, description
**Methods**: is_available, get_tools, execute, _send_email, _read_emails, _search_emails

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_email",
                "description": "Send an email",
                "risk": "confirm",
                "args": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "read_emails",
                "description": "Read recent emails from inbox",
                "risk": "low",
                "args": {
                    "folder": {"type": "string", "default": "INBOX"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required_args": [],
            },
            {
                "name": "search_emails",
                "description": "Search emails by keyword",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Subject keyword"},
                },
                "required_args": ["query"],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.