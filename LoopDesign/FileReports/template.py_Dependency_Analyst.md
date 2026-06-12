# File Report: template.py
## Role: Dependency Analyst

### 1. Library Requirements
- `typing` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- None.

### 3. Hidden Execution Links
- `is_available()` is hardcoded to return `False`. This ensures it is entirely skipped by `IntegrationLoader` during runtime registration.

### 4. Assumptions & API Contracts
- Serves purely as a documentation template/reference for developers building new plugins.
- Demonstrates how to handle unused parameters `del tool_name, args`.

### 5. Configuration Variables
- None.

### 6. Prompts Found
- None.
