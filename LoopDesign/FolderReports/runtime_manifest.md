# Runtime Folder Manifest

## Overview
The `runtime` directory is responsible for managing the active operational state, logs, ingestion data, and active session information for the Jarvis application. It acts as the dynamic state storage during execution.

## Contents

- **`automation_ingest.jsonl`** (File)
  - **Purpose**: Stores line-delimited JSON data for automation ingestion, likely acting as a queue or log of incoming data streams.
  - **Required Specialists**: Data Model Analyst, Runtime Investigator

- **`automation_state.json`** (File)
  - **Purpose**: Stores the current operational state of the automation system, keeping track of progress, locks, or active variables.
  - **Required Specialists**: Configuration Analyst, Runtime Investigator

- **`logs/`** (Directory)
  - **Purpose**: Stores runtime execution logs for diagnostics and debugging.
  - **Required Specialists**: Runtime Investigator
  - **`logs/jarvis.log`** (File)
    - **Purpose**: Main application log file capturing the runtime events, errors, and informational messages of the Jarvis system.
    - **Required Specialists**: Runtime Investigator

- **`sessions/`** (Directory)
  - **Purpose**: Directory intended to store active or historical user/system session data (currently empty).
  - **Required Specialists**: Data Model Analyst, Runtime Investigator
