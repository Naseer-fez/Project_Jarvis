# File Report: config/jarvis.ini

## Purpose
`jarvis.ini` is the primary configuration file for the Jarvis application. It defines system-wide settings, model selections, routing rules, safety constraints, automation configurations, multi-agent orchestrations, and integration details.

## Responsibilities
- Define model tiers (quick, chat, plan, vision) and provider endpoints (e.g., local Ollama URL).
- Set rules for risk levels and safety (forbidden actions, required user confirmations).
- Provide paths for memory storage (SQLite, Chroma, Goals).
- Control settings for features like web search, voice recognition, TTS, concurrency, logging, proactive alerting, automation directories (drop root, rag folders, etc.), and routing strategies.

## Architecture Role
This is the central source of truth for the entire application's behavior. Almost all subsystems (Memory, Action Execution, Routing, Agents, Multi-agent coordinator, Hardware, Web, Voice) pull their runtime parameters from this file.

## Dependencies & Interactions
- Provides settings for:
  - `[ollama]`, `[models]`: LLM interactions.
  - `[memory]`: Memory modules (SQLite, Chroma).
  - `[risk]`, `[execution]`: Action execution and safety modules.
  - `[web_search]`: Web search tools.
  - `[voice]`, `[hardware]`: External I/O and speech.
  - `[automation]`, `[multi_agent]`: Background watchers and multi-agent systems.
  - `[routing]`: Cost optimization and telemetry.

## Noteworthy Sections
- **Models**: Maps specific tasks to different tier models (e.g., `qwen2.5:0.5b` for intent, `mistral:7b` for tools, `deepseek-r1:8b` for planning).
- **Risk**: Explicit lists of high, medium, and low-risk tools/actions to enforce safety perimeters.
- **Automation**: Specifies dropbox folders (`workspace/jarvis_dropbox/*`) for event-driven file processing, screenshots, and screen recordings.
- **Multi-Agent**: Toggles specific subagents (interaction, web, desktop, rag, monitor).

## Prompts
- None. Pure configuration.
