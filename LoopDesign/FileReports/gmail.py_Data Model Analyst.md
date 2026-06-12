# File Report: gmail.py
**Path**: `d:\AI\Jarvis\integrations\clients\gmail.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- aiohttp
- base64
- os
- email.mime.text.MIMEText
- typing.Any
- integrations.base.BaseIntegration
- aiohttp
- aiohttp
- aiohttp
- aiohttp
- aiohttp

## Classes and State Objects
### `GmailIntegration`
**Variables**: name, description
**Methods**: is_available, get_tools, execute, _refresh_access_token, _list_unread, _get_message_meta, _summarize_unread, _send_gmail, _mark_as_read

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_unread",
                "description": "List unread emails from Gmail inbox",
                "risk": "low",
                "args": {
                    "max_results": {
                        "type": "integer",
                        "description": "Max emails to return",
                        "default": 10,
                    },
                },
                "required_args": [],
            },
            {
                "name": "send_gmail",
                "description": "Send an email via Gmail",
                "risk": "confirm",
                "args": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Plain-text email body"},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "summarize_unread",
                "description": "Fetch unread emails and return truncated content for LLM summarization",
                "risk": "low",
                "args": {
                    "max_results": {"type": "integer", "default": 5},
                },
                "required_args": [],
            },
            {
                "name": "mark_as_read",
                "description": "Mark a Gmail message as read by its message ID",
                "risk": "confirm",
                "args": {
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID",
                    },
                },
                "required_args": ["message_id"],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.