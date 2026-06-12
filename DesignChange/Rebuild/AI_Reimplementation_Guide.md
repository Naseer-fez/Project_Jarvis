# AI Reimplementation Guide

Instructions for an autonomous AI agent to rebuild the system from scratch.

## Step 1: Environment & Foundation
1. Initialize a Python 3.11+ repository.
2. Define `requirements.txt` based on the dependencies documented in `/DesignChange/Architecture/Infrastructure_Model.md`.
3. Create the `audit/` domain first. Implement a thread-safe `QueueListener` logger to guarantee all future components can inject logs.

## Step 2: The Core Interfaces
1. Scaffold `core/types` and `core/capability/base.py` to define the `ToolObservation` and `Capability` data models.
2. Build the `core.registry.registry.CapabilityRegistry` to allow runtime tool registration.
3. Build the `core.memory` interfaces (wrapping `aiosqlite` and `chromadb`). Do not implement the LLM orchestration yet.

## Step 3: Integrating the Tools
1. Scaffold the `integrations/` domain.
2. Implement dummy integrations for `Weather` and `Notion`. Register them via the loader to prove the dependency inversion works.

## Step 4: Autonomy and The Loop
1. Implement the LLM router (`core.llm`).
2. Implement `core.agent.agent_loop.AgentLoopEngine` to query the LLM router, parse the response into tool executions via the `CapabilityRegistry`, and log the result to `audit`.
3. Wrap this loop in `core.autonomy.goal_manager` to allow persistent tasking.

## Step 5: Web UI
1. Implement `dashboard/server.py` using `FastAPI`.
2. Connect endpoints to instantiate goals within the `GoalManager`.