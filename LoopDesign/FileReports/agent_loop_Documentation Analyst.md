# Analysis Report for agent_loop.py

## Dependencies
- __future__.annotations
- asyncio
- inspect
- logging
- re
- time
- dataclasses.dataclass
- dataclasses.field
- typing.Any
- typing.Optional
- core.state_machine.State
- core.state_machine.StateMachine
- core.context.context.TaskExecutionContext
- core.autonomy.autonomy_governor.AutonomyGovernor
- core.autonomy.risk_evaluator.RiskEvaluator
- core.planner.planner.TaskPlanner
- core.metrics.confidence.ConfidenceModel
- core.registry.registry.ToolObservation
- core.registry.registry.CapabilityRegistry

## Schemas
- ExecutionTrace
- ExecutionTrace attribute: goal
- ExecutionTrace attribute: iterations
- ExecutionTrace attribute: plan
- ExecutionTrace attribute: observations
- ExecutionTrace attribute: risk_scores
- ExecutionTrace attribute: think_blocks
- ExecutionTrace attribute: reflection
- ExecutionTrace attribute: final_response
- ExecutionTrace attribute: success
- ExecutionTrace attribute: stop_reason
- ExecutionTrace attribute: started_at
- ExecutionTrace attribute: ended_at
- AgentLoopEngine

## API Contracts
- _truncate_obs(text, max_chars)
- _truncate_observation(text, max_chars)
- ExecutionTrace.close(self, success, reason)
- ExecutionTrace.to_dict(self)
- AgentLoopEngine.__init__(self, state_machine, task_planner, tool_router, risk_evaluator, autonomy_governor, model, ollama_url, max_iterations, llm, container)
- AgentLoopEngine.request_interrupt(self)
- AgentLoopEngine._check_interrupt(self)
- AgentLoopEngine._ensure_thinking_state(self, sm)
- AgentLoopEngine._normalize_steps(self, plan)
- AgentLoopEngine._plan_summary(self, plan)
- AgentLoopEngine._fallback_reflection(self, plan, observations)
- AgentLoopEngine._stop(self, trace, reason, sm)

## Configuration Variables
- _DEFAULT_MAX_ITERATIONS
- REFLECT_SYSTEM_PROMPT

## Assumptions & Notes
- Module Docstring: Agent loop engine: plan -> risk -> confirm -> execute -> reflect.

