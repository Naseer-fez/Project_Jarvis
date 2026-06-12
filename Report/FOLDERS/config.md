# Folder Analysis: config

## Folder Purpose
Contains components related to config.

## Findings
- **JARVIS-CONFIG-001** (High): The configuration file references a blueprint file that does not exist in the config directory.
- **JARVIS-CONFIG-002** (Medium): The active environment file (`settings.env`) is severely out of sync with its template (`settings.env.template`), missing multiple critical configuration blocks.
- **JARVIS-CONFIG-003** (Low): Duplicated web search configuration across two different configuration domains (INI and ENV).
- **JARVIS-CONFIG-004** (Medium): Duplicate and redundant model configurations between the `[ollama]` and `[models]` sections.
- **JARVIS-CONFIG-005** (High): Conflicting risk categories. The `forbidden_actions`, `blocked_actions`, and `critical_actions` configuration keys are identically assigned the exact same list of dangerous actions.

## Risks & Dependencies
See full project roadmap.
