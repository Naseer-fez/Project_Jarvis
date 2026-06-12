# API Analyst Report: controller\intent_handlers.py

## Dependencies
- `import uuid`
- `import logging`
- `from typing import TYPE_CHECKING`
- `from core.controller.request_rules import is_active_window_request`
- `from core.controller.request_rules import is_explicit_web_search`
- `from core.desktop.shortcuts import handle_desktop_command`
- `from core.desktop.shortcuts import plan_desktop_command`
- `from core.controller.web_search import handle_web_search`

## Functions & Endpoints

### `register_intent_routes`
`def register_intent_routes(ctx: 'JarvisControllerV2') -> None`