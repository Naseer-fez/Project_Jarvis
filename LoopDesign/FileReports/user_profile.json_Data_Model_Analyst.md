# File Report: user_profile.json
**Role:** Data Model Analyst
**Target:** `d:\AI\Jarvis\memory\user_profile.json`

## Schema Overview
This file maintains the current state of the User Profile. It serves as a persistent state object detailing the system's understanding of the primary user.

### JSON Schema / DTO Mapping:
- `name` (string): The user's identified name or alias (e.g., `"hacked"`).
- `communication_style` (string): Preference or inferred style (e.g., `"casual"`).
- `expertise_level` (string): Assumed technical proficiency of the user (e.g., `"intermediate"`).
- `preferred_topics` (array of strings): List of topics the user frequently discusses or prefers.
- `timezone` (string): The user's local timezone (e.g., `"UTC"`).
- `language` (string): ISO language code (e.g., `"en"`).
- `interaction_count` (integer): Analytical metric tracking the number of interactions/sessions (e.g., `379`).
- `first_seen` (timestamp string): ISO-8601 formatted without timezone offset (e.g., `"2026-06-01T14:38:00.773283"`).
- `last_seen` (timestamp string): ISO-8601 formatted without timezone offset, updated on recent interaction.

## API Contracts & Dependencies
- Used to personalize agent responses (e.g., tailoring communication style, language, and expertise level).
- The `interaction_count` and `last_seen` imply a hook in the conversation/session lifecycle that updates this JSON file automatically upon user activity.

## State Objects & DTOs
- Root DTO representation:
  ```python
  class UserProfile:
      name: str
      communication_style: str
      expertise_level: str
      preferred_topics: list[str]
      timezone: str
      language: str
      interaction_count: int
      first_seen: datetime
      last_seen: datetime
  ```

## Configuration Variables & Prompts
- Configuration Variables:
  - `language`: Acts as a runtime config variable for localization.
  - `communication_style` / `expertise_level`: Act as dynamic prompt injection parameters.
- Prompts: No explicit raw prompts, but the fields directly feed into the System Prompt context for standardizing the LLM's persona/alignment to the user.
