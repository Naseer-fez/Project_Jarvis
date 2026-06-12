# API Analyst Report: desktop\shortcuts.py

## Dependencies
- `from __future__ import annotations`
- `import re`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from urllib.parse import quote_plus`
- `from core.desktop.actions import DesktopActionExecutor`
- `from core.desktop.contracts import DesktopAction`
- `from core.desktop.contracts import DesktopActionType`
- `from core.desktop.mission import DesktopMissionExecutor`
- `from core.desktop.mission import DesktopMissionStatus`
- `from core.desktop.observation import DesktopObserver`
- `from core.tools.system_automation import async_launch_application`

## Configuration Variables
- `INTERACTIVE_VERBS_PATTERN` = `re.compile('\\b(write|type|fill|enter|input|select|click|double-?click|right-?click|middle-?click|press|hotkey|key|drag|drop|scroll|typewrite|keystroke|tap|check|tick|focus|copy|paste|cut|clipboard|escribir|teclear|rellenar|introducir|pulsar|presionar|clic|pinchar|seleccionar|arrastrar|pegar|copiar|écrire|ecrire|taper|saisir|remplir|cliquer|appuyer|presser|sélectionner|selectionner|glisser|coller|copier|schreiben|tippen|eingeben|ausfüllen|ausfullen|klicken|drücken|drucken|auswählen|auswahlen|ziehen|einfügen|einfugen|kopieren)\\b', re.IGNORECASE)`
- `PROJECT_ROOT` = `Path(__file__).resolve().parent.parent.parent`

## Schemas & API Contracts (Classes)

### Class `DesktopCommandPlan`
**Fields/Schema:**
  - `app_label: str`
  - `primary_target: str`
  - `primary_args: list[str] | None`
  - `response_text: str`



## Functions & Endpoints

### `_supported_apps_message`
`def _supported_apps_message() -> str`
### `_extract_search_query`
`def _extract_search_query(lowered: str, original: str) -> str`
### `plan_desktop_command`
`def plan_desktop_command(user_input: str) -> DesktopCommandPlan | None`
### `handle_desktop_command`
`async def handle_desktop_command(user_input: str) -> str | None`