# Folder Analysis: tests

## Folder Purpose
Contains components related to tests.

## Findings
- **JARVIS-TESTS-001** (High): The `test_agent_loop_user_interrupt` test passes a synchronous lambda function (`lambda prompt: False`) as the `confirm_callback` argument to `engine.run()`. The `AgentLoopEngine` expects this callback to be an asynchronous function, as it is awaited internally.

## Risks & Dependencies
See full project roadmap.
