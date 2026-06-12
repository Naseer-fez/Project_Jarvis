ISSUE ID: JARVIS-CONFIG-001
SEVERITY: High
CATEGORY: Missing implementations
FILES: d:\AI\Jarvis\config\jarvis.ini
DESCRIPTION: The configuration file references a blueprint file that does not exist in the config directory.
ROOT CAUSE: The `[ai_os]` section sets `blueprint_file = config/ai_os.json`, but `ai_os.json` is completely missing from the `config` folder.
EVIDENCE: 
Line 125 in jarvis.ini: `blueprint_file = config/ai_os.json`
The directory listing of `d:\AI\Jarvis\config` only contains `jarvis.ini`, `settings.env`, and `settings.env.template`.
POTENTIAL IMPACT: Will cause a FileNotFoundError or application crash when the AI OS module attempts to read its blueprint file upon initialization.
RECOMMENDED FIX: Create the missing `ai_os.json` file in the `config` directory with the necessary blueprint schema, or update the `blueprint_file` path to point to a valid existing file.

ISSUE ID: JARVIS-CONFIG-002
SEVERITY: Medium
CATEGORY: Configuration problems
FILES: d:\AI\Jarvis\config\settings.env, d:\AI\Jarvis\config\settings.env.template
DESCRIPTION: The active environment file (`settings.env`) is severely out of sync with its template (`settings.env.template`), missing multiple critical configuration blocks.
ROOT CAUSE: `settings.env` was likely not updated when new integrations (Home Assistant, GitHub, Quick local model routing) were added to the `.template` file.
EVIDENCE: 
`settings.env.template` contains `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`, `GITHUB_TOKEN`, `GITHUB_DEFAULT_REPO`, and various `WEB_SEARCH_*` keys (lines 13-42) which are entirely absent from `settings.env` (which only goes up to line 21).
POTENTIAL IMPACT: Attempting to use Home Assistant or GitHub features will fail because the required credentials and URLs are absent from the active environment variables, leading to broken integrations.
RECOMMENDED FIX: Update `settings.env` to include all the missing keys from `settings.env.template` so the user can properly configure them.

ISSUE ID: JARVIS-CONFIG-003
SEVERITY: Low
CATEGORY: Architectural inconsistencies
FILES: d:\AI\Jarvis\config\jarvis.ini, d:\AI\Jarvis\config\settings.env.template
DESCRIPTION: Duplicated web search configuration across two different configuration domains (INI and ENV).
ROOT CAUSE: Web search settings are defined in the `[web_search]` section of `jarvis.ini` and redundantly specified as `WEB_SEARCH_*` environment variables in `settings.env.template`.
EVIDENCE:
`jarvis.ini` (Lines 67-79):
[web_search]
enabled = true
provider = auto
default_max_results = 5

`settings.env.template` (Lines 31-42):
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=auto
WEB_SEARCH_DEFAULT_MAX_RESULTS=5
POTENTIAL IMPACT: Causes configuration drift and confusion. Modifying settings in one file might have no effect if the application prioritizes the other, leading to frustrating debugging experiences for developers.
RECOMMENDED FIX: Remove the redundant `WEB_SEARCH_*` variables from `settings.env.template` and consolidate all web search configurations strictly within `jarvis.ini`.

ISSUE ID: JARVIS-CONFIG-004
SEVERITY: Medium
CATEGORY: Configuration problems
FILES: d:\AI\Jarvis\config\jarvis.ini
DESCRIPTION: Duplicate and redundant model configurations between the `[ollama]` and `[models]` sections.
ROOT CAUSE: The models `vision_model = llava` and `plan(ner)_model = deepseek-r1:8b` are configured redundantly in both the `[ollama]` block and the general `[models]` block.
EVIDENCE:
Lines 8-9 (in `[ollama]`): 
planner_model = deepseek-r1:8b
vision_model = llava
Lines 23-25 (in `[models]`): 
plan_model = deepseek-r1:8b
vision_model = llava
POTENTIAL IMPACT: Changing a model in one section may not take effect if the system reads from the other section, leading to the wrong models being executed for tasks and causing unexpected behaviors.
RECOMMENDED FIX: Consolidate model configurations. Remove model specifications from `[ollama]` and rely solely on the `[models]` section, or explicitly define/document which block takes precedence in the application logic.

ISSUE ID: JARVIS-CONFIG-005
SEVERITY: High
CATEGORY: Security vulnerabilities
FILES: d:\AI\Jarvis\config\jarvis.ini
DESCRIPTION: Conflicting risk categories. The `forbidden_actions`, `blocked_actions`, and `critical_actions` configuration keys are identically assigned the exact same list of dangerous actions.
ROOT CAUSE: A copy-paste error during the creation of the `[risk]` configuration section.
EVIDENCE:
Lines 41-43:
forbidden_actions = execute_shell, delete_file, shell_exec, file_delete, registry_write, format_disk, wipe_disk
blocked_actions = execute_shell, delete_file, shell_exec, file_delete, registry_write, format_disk, wipe_disk
critical_actions = execute_shell, delete_file, shell_exec, file_delete, registry_write, format_disk, wipe_disk
POTENTIAL IMPACT: Critical security risk. If the permissions logic checks `critical_actions` before `forbidden_actions`, an attacker or runaway agent could execute a forbidden action (like `format_disk` or `execute_shell`) simply by passing/bypassing the "critical" user confirmation prompt.
RECOMMENDED FIX: Segregate the actions appropriately. Truly dangerous actions should only exist in `forbidden_actions`, while actions requiring user consent should only exist in `critical_actions`. Remove the duplication across the three arrays.
