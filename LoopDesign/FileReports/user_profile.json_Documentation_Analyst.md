# File Report: user_profile.json
**Role**: Documentation Analyst
**Target**: `d:\AI\Jarvis\memory\user_profile.json`

## Overview
JSON document used to persist the user's profile, preferences, and interaction statistics. Serves as a long-term context anchor for the agent's personalization.

## Schema & API Contract
- `name`: String. The identifier or name of the user (e.g., `"hacked"`).
- `communication_style`: String. Guides the agent's tone (e.g., `"casual"`).
- `expertise_level`: String. Indicates the user's knowledge depth, tailoring responses (e.g., `"intermediate"`).
- `preferred_topics`: Array. List of topics of interest to the user.
- `timezone`: String. User's local timezone (e.g., `"UTC"`).
- `language`: String. Language preference code (e.g., `"en"`).
- `interaction_count`: Integer. Counter tracking the total number of interactions with the user.
- `first_seen`: String. ISO-8601 formatted datetime without timezone offset.
- `last_seen`: String. ISO-8601 formatted datetime without timezone offset.

## Assumptions & Design Patterns
- **Personalization Anchor**: This data is likely injected into the main agent prompt to condition tone, verbosity, and complexity.
- **Stat Tracking**: The application updates this file iteratively (indicated by `interaction_count` and `last_seen`).
- **Naive Datetimes**: In contrast to `goals.json`, the datetimes here (`first_seen`, `last_seen`) lack timezone offsets, indicating potentially inconsistent datetime handling across different memory components (naive vs timezone-aware).

## Developer Notes
- This schema acts as a direct configuration variable set for agent persona matching.
