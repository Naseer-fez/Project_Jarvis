# Folder Manifest: memory

## High-Level Purpose
The `memory` directory serves as the persistent data storage layer for the system. It contains structured data files for tracking user goals, schedules, and profile information, alongside SQLite database files used for long-term memory and runtime state management. 

## Contents and Required Specialists

### Files

- **`goals.json`**
  - **Description**: Stores user goals and scheduled tasks in JSON format.
  - **Required Specialists**: Data Model Analyst, Configuration Analyst.

- **`memory.db`**
  - **Description**: SQLite database used for storing and retrieving long-term memory entries and complex structured data.
  - **Required Specialists**: Data Model Analyst, Runtime Investigator.

- **`memory.db-shm`**
  - **Description**: SQLite shared memory file (used for WAL mode).
  - **Required Specialists**: Data Model Analyst.

- **`memory.db-wal`**
  - **Description**: SQLite write-ahead log file containing uncommitted changes.
  - **Required Specialists**: Data Model Analyst.

- **`user_profile.json`**
  - **Description**: JSON file containing user profile details, interaction counts, timezone, and communication style preferences.
  - **Required Specialists**: Data Model Analyst, Configuration Analyst.

### Subdirectories
- *(None)*
