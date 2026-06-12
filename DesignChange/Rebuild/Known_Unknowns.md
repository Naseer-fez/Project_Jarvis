# Known Unknowns

While the static analysis maps the structure accurately, certain dynamic properties are missing from the documentation and must be empirically derived or safely defaulted during the rebuild:

- **Third-Party API Rate Limits**: The exact retry delays and backoff thresholds for external services (e.g. `github`, `gmail`) are not hardcoded in the AST analysis. Safe defaults (e.g., exponential backoff) should be implemented.
- **LLM Prompt Schemas**: The precise system prompts injected by `core.planner.planner.TaskPlanner` and `core.profile.UserProfileEngine` are mostly textual strings and require empirical tuning during the re-implementation.
- **Hardware Bindings**: Serial baud rates and specific `pvporcupine` wake-word sensitivity thresholds depend on the physical deployment hardware.
- **Desktop Coordinate Boundaries**: The `DesktopObserver` and `DesktopActionExecutor` rely heavily on the local OS resolution and screen layout which cannot be generalized.