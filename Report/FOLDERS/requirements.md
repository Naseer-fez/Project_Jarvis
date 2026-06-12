# Folder Analysis: requirements

## Folder Purpose
Contains components related to requirements.

## Findings
- **REQUIREMENTS-001** (Medium): Use of the deprecated and unmaintained `fpdf` library.
- **REQUIREMENTS-002** (Low): Type stubs for `PyYAML` are included in development requirements, but the actual runtime library `PyYAML` is missing from all requirements files.
- **REQUIREMENTS-003** (Low): The development environment configuration does not include the full application stack, leading to dangling dependencies and broken dev setups.

## Risks & Dependencies
See full project roadmap.
