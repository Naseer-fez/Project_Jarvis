# File Report: all_clients.txt
## Role: Dependency Analyst

### 1. Library Requirements
- Matches all libraries found in the individual `clients/*.py` files.

### 2. Service Dependencies
- Matches all external service dependencies (Google APIs, GitHub, Spotify, Notion, Home Assistant, etc.).

### 3. Hidden Execution Links
- None. This is a text file containing concatenated source code of all the integration python files in `clients/`. It is not actively loaded by `loader.py` (which only looks for `*.py` files).

### 4. Assumptions & API Contracts
- Used as a reference or backup of integration code.

### 5. Configuration Variables
- Contains all required variables specified in the respective integrations.

### 6. Prompts Found
- None.
