ISSUE ID: DesignCHnage-001
SEVERITY: Medium
CATEGORY: Error handling gaps
FILES: d:\AI\Jarvis\DesignCHnage\investigate_domain.py
DESCRIPTION: Unhandled `FileNotFoundError` in the `update_ledger()` function.
ROOT CAUSE: The `update_ledger` function assumes `COVERAGE_LEDGER_PATH` exists without checking, whereas the `parse_ledger` function explicitly handles its absence gracefully. If the file is missing, the script will crash inside `update_ledger()`.
EVIDENCE: 
```python
def update_ledger():
    with open(COVERAGE_LEDGER_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
```
POTENTIAL IMPACT: Script fails with an exception if the ledger file is deleted or renamed, halting execution.
RECOMMENDED FIX: Add a file existence check `if not COVERAGE_LEDGER_PATH.exists(): return` inside `update_ledger()`.

ISSUE ID: DesignCHnage-002
SEVERITY: Low
CATEGORY: Documentation mismatches
FILES: d:\AI\Jarvis\DesignCHnage\DesignCHnage.md
DESCRIPTION: Incorrect file paths in markdown links.
ROOT CAUSE: The markdown links in the Final Deliverables section point to `file:///d:/AI/Jarvis/` instead of `file:///d:/AI/Jarvis/DesignCHnage/` where the actual generated files are stored.
EVIDENCE:
`1. [Repository_Census.md](file:///d:/AI/Jarvis/Repository_Census.md)`
POTENTIAL IMPACT: Users clicking the links will encounter File Not Found errors.
RECOMMENDED FIX: Update the file paths in `DesignCHnage.md` to point to the correct subdirectory: `file:///d:/AI/Jarvis/DesignCHnage/`.

ISSUE ID: DesignCHnage-003
SEVERITY: High
CATEGORY: Logic errors
FILES: d:\AI\Jarvis\DesignCHnage\orchestrate_recovery.py
DESCRIPTION: The script unconditionally overwrites existing detailed architecture, validation, and rebuild documentation files with short hardcoded stubs.
ROOT CAUSE: In `orchestrate_recovery.py`, the `open()` function is used with mode `"w"` unconditionally for all markdown files. Since the actual documentation files (e.g. `Architecture_Overview.md` which is ~1800 bytes) contain detailed content, running this script truncates and overwrites them with the short stub strings defined in the script's dictionaries.
EVIDENCE:
```python
    for filename, content in files.items():
        with open(ARCHITECTURE_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)
```
POTENTIAL IMPACT: Severe data loss; running the script inadvertently destroys rich, previously generated documentation artifacts across four directories.
RECOMMENDED FIX: Use mode `"x"` to prevent overwriting, or explicitly check file existence before opening in write mode.

ISSUE ID: DesignCHnage-004
SEVERITY: Low
CATEGORY: Data validation issues
FILES: d:\AI\Jarvis\DesignCHnage\Investigation_Plan.md
DESCRIPTION: Truncated dependency lists causing broken package references.
ROOT CAUSE: The dependency strings for multiple batches are arbitrarily cut off (e.g., `s` instead of `speech_recognition`, `teleg` instead of `telegram`, `configpar` instead of `configparser`). This indicates that whatever script generated this file applied a hard character limit or naive substring operation, leading to invalid dependency names.
EVIDENCE:
`Batch_02: ... fpdf, textwrap, s`
`Batch_03: ... base64, teleg`
`Batch_05: ... inspect, configpar`
POTENTIAL IMPACT: Downstream tools parsing these dependencies may attempt to install or evaluate non-existent packages, causing setup or runtime failures.
RECOMMENDED FIX: Re-generate the `Investigation_Plan.md` file without hard-coded substring limits, allowing the list to wrap naturally.

ISSUE ID: DesignCHnage-005
SEVERITY: Low
CATEGORY: Architectural inconsistencies
FILES: d:\AI\Jarvis\DesignCHnage\investigate_domain.py
DESCRIPTION: Non-deterministic output in `generate_data_flow` due to unordered set slicing.
ROOT CAUSE: In `generate_data_flow`, `deps` is a `set()`. `list(deps)[:5]` extracts the first 5 elements. Since Python sets do not preserve order, and string hashing seeds change per execution, the generated Markdown will randomly list different dependencies on every run.
EVIDENCE:
```python
    deps = set()
    ...
    content += f"Data exchanges primarily with: {', '.join(list(deps)[:5])}\n"
```
POTENTIAL IMPACT: Unstable generation of `Data_Flow.md` artifacts, causing unnecessary diffs and confusion for reviewers or agents processing changes.
RECOMMENDED FIX: Sort the set before slicing: `', '.join(sorted(list(deps))[:5])`.

ISSUE ID: DesignCHnage-006
SEVERITY: Low
CATEGORY: Logic errors
FILES: d:\AI\Jarvis\DesignCHnage\investigate_domain.py
DESCRIPTION: Incorrect counting of top-level functions in evidence reports.
ROOT CAUSE: The `Analyzer` class iterates through AST nodes. `visit_FunctionDef` appends all functions to `self.functions`. Because `visit_ClassDef` calls `self.generic_visit`, class methods are also captured and grouped with top-level functions. The generated evidence explicitly documents this total count as "top-level functions", which is factually incorrect.
EVIDENCE:
```python
# In generate_evidence:
content += f"- **{file}**: Verified {len(data['classes'])} classes and {len(data['functions'])} top-level functions.\n"
```
POTENTIAL IMPACT: Inaccurate structural metrics in the evidence reports, leading to confusion during architectural review.
RECOMMENDED FIX: Either implement depth tracking in the AST visitor to separate top-level functions from methods, or rename the report metric to "total functions".

ISSUE ID: DesignCHnage-007
SEVERITY: Low
CATEGORY: Configuration problems
FILES: d:\AI\Jarvis\DesignCHnage
DESCRIPTION: Obvious typo in the root folder name and corresponding main markdown file.
ROOT CAUSE: A typographical error was made when naming `DesignCHnage` (Capital 'H', lowercase 'n') instead of `DesignChange`.
EVIDENCE: The directory itself and the root file `DesignCHnage.md` reflect this. Both scripts map to `DESIGN_CHANGE_DIR = ROOT_DIR / "DesignCHnage"`.
POTENTIAL IMPACT: Integration confusion and risk of path-related errors if external agents dynamically attempt to reconstruct the standard `DesignChange` path name.
RECOMMENDED FIX: Rename the folder and the markdown file to `DesignChange`, and update the scripts correspondingly.
