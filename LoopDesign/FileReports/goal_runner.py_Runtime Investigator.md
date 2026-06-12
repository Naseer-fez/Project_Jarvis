# Runtime Investigator Report: goal_runner.py

## Role Relevancy
Handles persistent runtime task monitoring and scheduled triggering of user intents/goals.

## Assumptions
- Saves and restores state to a persistent `goals_file` via `json.dumps`.
- Runs a daemon thread loop checking scheduler timeouts against `goal_check_interval_seconds`.
- Dispatches TTS speech routines via `voice_layer` upon goal completion/activation.

## Schema & API Contracts
- `GoalRunner`: `.load_goal_state()`, `.persist_goal_state()`, `.check_due_goals()`.

## Dependencies
- Interacts with abstract `goal_manager`, `scheduler`, `notifier`, and `voice_layer`.

## Configuration Variables
- Driven by passed `goal_check_interval_seconds` and `goals_file` Path args.

## Prompts
- None.
