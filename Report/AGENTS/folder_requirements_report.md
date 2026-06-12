ISSUE ID: REQUIREMENTS-001
SEVERITY: Medium
CATEGORY: Dependency issues
FILES: d:\AI\Jarvis\requirements\full.txt
DESCRIPTION: Use of the deprecated and unmaintained `fpdf` library.
ROOT CAUSE: The original `fpdf` package has been abandoned for years and does not support modern Python environments. The officially maintained successor is `fpdf2`.
EVIDENCE: `fpdf>=1.7,<2.0` is specified in `full.txt`.
POTENTIAL IMPACT: Lack of security patches, missing modern features (like proper UTF-8 support), and potential compatibility failures with newer Python versions.
RECOMMENDED FIX: Replace `fpdf>=1.7,<2.0` with `fpdf2>=2.8.0` (or the latest stable version) in `full.txt`, and verify `fpdf` imports in the codebase.

ISSUE ID: REQUIREMENTS-002
SEVERITY: Low
CATEGORY: Dependency issues
FILES: d:\AI\Jarvis\requirements\dev.txt, d:\AI\Jarvis\requirements\base.txt
DESCRIPTION: Type stubs for `PyYAML` are included in development requirements, but the actual runtime library `PyYAML` is missing from all requirements files.
ROOT CAUSE: Either the runtime dependency `PyYAML` was forgotten and the app relies on a transitive dependency (which is fragile), or the type stubs are dead/unused code.
EVIDENCE: `types-PyYAML>=6.0,<7.0` is present in `dev.txt`, yet `PyYAML` is completely absent from `base.txt`, `full.txt`, and all other files.
POTENTIAL IMPACT: Potential runtime crashes (`ModuleNotFoundError`) if the codebase attempts to import `yaml` in production and the dependency is not implicitly installed, or unnecessary bloat in the dev environment.
RECOMMENDED FIX: Explicitly add `PyYAML>=6.0` to `base.txt` or `full.txt` if YAML parsing is used, or remove `types-PyYAML` from `dev.txt` if not.

ISSUE ID: REQUIREMENTS-003
SEVERITY: Low
CATEGORY: Architectural inconsistencies
FILES: d:\AI\Jarvis\requirements\dev.txt, d:\AI\Jarvis\requirements\full.txt
DESCRIPTION: The development environment configuration does not include the full application stack, leading to dangling dependencies and broken dev setups.
ROOT CAUSE: `dev.txt` is based only on `base.txt` but includes type stubs intended for the full stack (e.g., `types-Markdown`). However, the actual runtime dependency (`markdown`) is isolated in `full.txt`.
EVIDENCE: `dev.txt` contains `-r base.txt` and `types-Markdown>=3.10,<4.0`. `full.txt` contains `markdown>=3.10,<4.0`. Installing `dev.txt` will install type stubs for markdown without the library itself, breaking static type checking and making it impossible to test full stack features.
POTENTIAL IMPACT: Developers running type checkers (like mypy) or executing integration tests will encounter missing module errors for features that rely on the full stack.
RECOMMENDED FIX: Change `dev.txt` to include `-r full.txt` instead of `-r base.txt`, ensuring the development and testing environment has all dependencies required to validate the complete application.
