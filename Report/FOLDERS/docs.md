# Folder Analysis: docs

## Folder Purpose
Contains components related to docs.

## Findings
- **DOCS-001** (Medium): The design document embeds images using hardcoded, absolute local file URIs that point to a temporary AI workspace, which will be broken for any other user or system.
- **DOCS-002** (Low): The "Codebase Directory Hierarchy" section is outdated and missing several directories and files that exist in the actual `core/` module.
- **DOCS-003** (Low): There is a Mermaid diagram syntax error and broken linkage in the "System Bootstrapping Flow" flowchart due to a mismatch between a subgraph ID and its reference.

## Risks & Dependencies
See full project roadmap.
