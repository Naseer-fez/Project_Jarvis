# Manifest: `config` Folder

**High-Level Purpose**:
This directory serves as the centralized repository for environment settings, system configurations, and application-level parameters for Project Jarvis. It dictates how the system initializes, authenticates with external services, and defines structural metadata for the AI operating system.

**File and Subfolder Manifest**:

| Item | Type | Required Tier 2 Specialists | Description / Purpose |
|------|------|-----------------------------|-----------------------|
| `ai_os.json` | File | Configuration Analyst, Data Model Analyst | Likely contains structured configurations or schema definitions mapping out the AI operating system's core capabilities or state definitions. |
| `jarvis.ini` | File | Configuration Analyst | Standard initialization file for general application settings, base system paths, or runtime flags. |
| `settings.env` | File | Configuration Analyst, API Analyst | Environment variables, including API keys, database URLs, and application secrets. |
| `settings.env.template` | File | Configuration Analyst | Boilerplate/template file defining the required environment variables without exposing actual secrets, used for initial setup. |
