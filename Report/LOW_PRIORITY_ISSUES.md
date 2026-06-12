# Low Priority Issues

### AUDIT-005 - Configuration problems
**Files:** d:\AI\Jarvis\audit\edge_cases\bad_config.ini
**Description:** The configuration file contains malformed INI syntax and invalid configuration types.
**Root Cause:** Line 1 is missing a closing bracket (`]`) for the section header, and line 2 assigns a non-numeric string (`invalid`) to a port variable.
**Impact:** If processed, `configparser` will raise a `MissingSectionHeaderError` or a `ValueError`, causing application startup crashes.
**Fix:** 

### AUDIT-006 - Database issues
**Files:** d:\AI\Jarvis\audit\edge_cases\memory\memory.db
**Description:** The file intended to act as an SQLite database (`memory.db`) is corrupted or invalid, measuring only 36 bytes with a text payload rather than a valid SQLite header.
**Root Cause:** An SQLite database requires a standard 100-byte header starting with "SQLite format 3\000". A 36-byte file lacks the valid SQLite database structure.
**Impact:** Any SQLite driver attempting to connect to or query this database file will throw a `DatabaseError: file is not a database`, resulting in database connection failures.
**Fix:** 

### AUDIT-007 - Configuration problems
**Files:** d:\AI\Jarvis\audit\edge_cases\test_config.ini
**Description:** The `sqlite_file` path is specified as a relative file path, making resolution fragile.
**Root Cause:** Hardcoding relative paths depends strictly on the application's current working directory matching the project root at runtime.
**Impact:** Running the application from any directory other than the root will cause file resolution failures or lead to an unintended database being created in the wrong directory.
**Fix:** 

### JARVIS-CONFIG-003 - Architectural inconsistencies
**Files:** d:\AI\Jarvis\config\jarvis.ini, d:\AI\Jarvis\config\settings.env.template
**Description:** Duplicated web search configuration across two different configuration domains (INI and ENV).
**Root Cause:** Web search settings are defined in the `[web_search]` section of `jarvis.ini` and redundantly specified as `WEB_SEARCH_*` environment variables in `settings.env.template`.
**Impact:** Causes configuration drift and confusion. Modifying settings in one file might have no effect if the application prioritizes the other, leading to frustrating debugging experiences for developers.
**Fix:** 

### JARVIS-CORE-004 - Error handling gaps
**Files:** `d:\AI\Jarvis\core\controller_v2.py`
**Description:** The CLI execution loop does not wrap the invocation of `await self.process(text)` in a `try/except` block.
**Root Cause:** Assuming `process()` gracefully handles all exceptions internally, which is rarely true across a wide subsystem suite (e.g. LLM request timeouts, invalid DAG plans).
**Impact:** An unexpected network error or data payload issue will crash the entire CLI application unexpectedly.
**Fix:** 

### DesignCHnage-002 - Documentation mismatches
**Files:** d:\AI\Jarvis\DesignCHnage\DesignCHnage.md
**Description:** Incorrect file paths in markdown links.
**Root Cause:** The markdown links in the Final Deliverables section point to `file:///d:/AI/Jarvis/` instead of `file:///d:/AI/Jarvis/DesignCHnage/` where the actual generated files are stored.
**Impact:** Users clicking the links will encounter File Not Found errors.
**Fix:** 

### DesignCHnage-004 - Data validation issues
**Files:** d:\AI\Jarvis\DesignCHnage\Investigation_Plan.md
**Description:** Truncated dependency lists causing broken package references.
**Root Cause:** The dependency strings for multiple batches are arbitrarily cut off (e.g., `s` instead of `speech_recognition`, `teleg` instead of `telegram`, `configpar` instead of `configparser`). This indicates that whatever script generated this file applied a hard character limit or naive substring operation, leading to invalid dependency names.
**Impact:** Downstream tools parsing these dependencies may attempt to install or evaluate non-existent packages, causing setup or runtime failures.
**Fix:** 

### DesignCHnage-005 - Architectural inconsistencies
**Files:** d:\AI\Jarvis\DesignCHnage\investigate_domain.py
**Description:** Non-deterministic output in `generate_data_flow` due to unordered set slicing.
**Root Cause:** In `generate_data_flow`, `deps` is a `set()`. `list(deps)[:5]` extracts the first 5 elements. Since Python sets do not preserve order, and string hashing seeds change per execution, the generated Markdown will randomly list different dependencies on every run.
**Impact:** Unstable generation of `Data_Flow.md` artifacts, causing unnecessary diffs and confusion for reviewers or agents processing changes.
**Fix:** 

### DesignCHnage-006 - Logic errors
**Files:** d:\AI\Jarvis\DesignCHnage\investigate_domain.py
**Description:** Incorrect counting of top-level functions in evidence reports.
**Root Cause:** The `Analyzer` class iterates through AST nodes. `visit_FunctionDef` appends all functions to `self.functions`. Because `visit_ClassDef` calls `self.generic_visit`, class methods are also captured and grouped with top-level functions. The generated evidence explicitly documents this total count as "top-level functions", which is factually incorrect.
**Impact:** Inaccurate structural metrics in the evidence reports, leading to confusion during architectural review.
**Fix:** 

### DesignCHnage-007 - Configuration problems
**Files:** d:\AI\Jarvis\DesignCHnage
**Description:** Obvious typo in the root folder name and corresponding main markdown file.
**Root Cause:** A typographical error was made when naming `DesignCHnage` (Capital 'H', lowercase 'n') instead of `DesignChange`.
**Impact:** Integration confusion and risk of path-related errors if external agents dynamically attempt to reconstruct the standard `DesignChange` path name.
**Fix:** 

### DOCS-002 - Documentation mismatches
**Files:** d:\AI\Jarvis\docs\design_doc.md
**Description:** The "Codebase Directory Hierarchy" section is outdated and missing several directories and files that exist in the actual `core/` module.
**Root Cause:** The architecture document's directory tree block was not updated to reflect recent additions and refactoring within the `core/` module.
**Impact:** New developers may experience confusion due to discrepancies between the documentation and the actual repository structure, hindering onboarding.
**Fix:** 

### DOCS-003 - Syntax issues
**Files:** d:\AI\Jarvis\docs\design_doc.md
**Description:** There is a Mermaid diagram syntax error and broken linkage in the "System Bootstrapping Flow" flowchart due to a mismatch between a subgraph ID and its reference.
**Root Cause:** The subgraph is declared with an ID containing a space (`subgraph Service Pool`), but it is referenced later without the space (`Controller --> ServicePool`).
**Impact:** The architecture diagram may fail to render in standard markdown viewers, or it will render incorrectly by creating a disconnected node named "ServicePool" instead of linking to the subgraph.
**Fix:** 

### FINAL-003 - Syntax issues
**Files:** d:\AI\Jarvis\Final\Architecture_Map.md
**Description:** There is a typographical spelling error in the section header describing the Dashboard component.
**Root Cause:** Human typo or automated agent generation error during markdown creation.
**Impact:** Minimal runtime impact, but it degrades documentation quality and could disrupt automated string-matching tools parsing the documentation.
**Fix:** 

### INTEGRATIONS-006 - Error handling gaps
**Files:** `integrations/clients/gmail.py`
**Description:** In `_get_message_meta()`, if the HTTP request to fetch individual message metadata fails (e.g., due to a rate-limiting 429 or 404), the method silently parses the error payload and returns an empty dictionary.
**Root Cause:** The method lacks an explicit check for `resp.status == 200` before accessing payload fields.
**Impact:** Users receive empty or blank summaries for unread emails without any indication that a background API error occurred.
**Fix:** 

### REQUIREMENTS-002 - Dependency issues
**Files:** d:\AI\Jarvis\requirements\dev.txt, d:\AI\Jarvis\requirements\base.txt
**Description:** Type stubs for `PyYAML` are included in development requirements, but the actual runtime library `PyYAML` is missing from all requirements files.
**Root Cause:** Either the runtime dependency `PyYAML` was forgotten and the app relies on a transitive dependency (which is fragile), or the type stubs are dead/unused code.
**Impact:** Potential runtime crashes (`ModuleNotFoundError`) if the codebase attempts to import `yaml` in production and the dependency is not implicitly installed, or unnecessary bloat in the dev environment.
**Fix:** 

### REQUIREMENTS-003 - Architectural inconsistencies
**Files:** d:\AI\Jarvis\requirements\dev.txt, d:\AI\Jarvis\requirements\full.txt
**Description:** The development environment configuration does not include the full application stack, leading to dangling dependencies and broken dev setups.
**Root Cause:** `dev.txt` is based only on `base.txt` but includes type stubs intended for the full stack (e.g., `types-Markdown`). However, the actual runtime dependency (`markdown`) is isolated in `full.txt`.
**Impact:** Developers running type checkers (like mypy) or executing integration tests will encounter missing module errors for features that rely on the full stack.
**Fix:** 

### JARVIS-RUNTIME-001 - Configuration problems
**Files:** d:\AI\Jarvis\runtime\logs\jarvis.log
**Description:** The application is sending unauthenticated requests to the Hugging Face Hub, which may result in lower rate limits and slower downloads.
**Root Cause:** The `HF_TOKEN` environment variable or authentication token is not configured in the environment where Jarvis is running.
**Impact:** The application might face rate limiting (HTTP 429 errors) or slower download speeds when fetching models from the Hugging Face Hub, potentially delaying initialization or causing runtime failures if rate limits are eventually exceeded.
**Fix:** 

### 6 - Dependencies between folders
**Files:** audit/audit_logger.py, core/logging/logger.py
**Description:** The `audit` directory is orphaned. It contains secret scrubbing logic (`audit_logger.py`) that is disconnected from the active runtime. The actual logging system implements its own redundant data redaction.
**Root Cause:** Architectural divergence where logging was rebuilt and consolidated into `core.logging`, abandoning the top-level `audit` module without deprecation.
**Impact:** Accumulation of dead code, confusion for maintainers, and security risks if engineers attempt to patch the orphaned `audit` module instead of the active `core` module.
**Fix:** 

