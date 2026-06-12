ISSUE ID: JARVIS-TESTS-001
SEVERITY: High
CATEGORY: Broken tests
FILES: d:\AI\Jarvis\tests\unit\test_agent_loop.py
DESCRIPTION: The `test_agent_loop_user_interrupt` test passes a synchronous lambda function (`lambda prompt: False`) as the `confirm_callback` argument to `engine.run()`. The `AgentLoopEngine` expects this callback to be an asynchronous function, as it is awaited internally.
ROOT CAUSE: A synchronous lambda was provided where an asynchronous callable is required. This results in a runtime `TypeError` (`object bool can't be used in 'await' expression`) when the callback is awaited.
EVIDENCE:
```python
# In d:\AI\Jarvis\tests\unit\test_agent_loop.py (line 92)
# Simulate user rejecting the prompt
trace = await engine.run("test goal", context, confirm_callback=lambda prompt: False)
```
POTENTIAL IMPACT: The test will consistently fail at runtime, blocking CI/CD pipelines and preventing the validation of user interrupt logic.
RECOMMENDED FIX: Replace the lambda with an asynchronous mock or helper function. For example:
```python
async def mock_confirm(prompt): return False
trace = await engine.run("test goal", context, confirm_callback=mock_confirm)
```
