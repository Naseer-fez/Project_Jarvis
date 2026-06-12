# API Analyst Report: proactive\notifier.py

## Dependencies
- `import logging`
- `import time`

## Schemas & API Contracts (Classes)

### Class `NotificationManager`
**Methods:**
- `def notify(self, message: str, level: str='info', voice_layer=None) -> None`
- `def schedule_reminder(self, message: str, in_seconds: int) -> None`
- `async def _delayed_notify(self, message: str, delay: int) -> None`

