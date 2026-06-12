# Folder Analysis: logs

## Folder Purpose
Contains components related to logs.

## Findings
- **LOGS-001** (High): The application logs indicate missing dependencies for critical voice components: `faster-whisper` (STT), `piper-tts` (TTS), and `openwakeword` (Wake Word). The application is forced to fallback to dummy or CLI implementations.
- **LOGS-002** (High): The application repeatedly fails to connect to Ollama and cloud LLM providers, leading to timeout errors and planner degradation.
- **LOGS-003** (Medium): The audit log file is marked as corrupted, indicating a failure to properly write or close the JSONL audit file.

## Risks & Dependencies
See full project roadmap.
