ISSUE ID: LOGS-001
SEVERITY: High
CATEGORY: Dependency issues
FILES: d:\AI\Jarvis\logs\app.log
DESCRIPTION: The application logs indicate missing dependencies for critical voice components: `faster-whisper` (STT), `piper-tts` (TTS), and `openwakeword` (Wake Word). The application is forced to fallback to dummy or CLI implementations.
ROOT CAUSE: The Python environment running Jarvis is missing required packages and model files (e.g., `en_US-lessac-medium.json`, `alexa_v0.1.onnx`).
EVIDENCE: `2026-02-20 16:41:40,012 ERROR jarvis.voice.stt: faster-whisper not installed — STT unavailable`
POTENTIAL IMPACT: Voice interaction is completely disabled or using unusable mock fallbacks, breaking voice-related features.
RECOMMENDED FIX: Install the missing dependencies (`faster-whisper`, `piper-tts`, `openwakeword`) and ensure the required model weights are downloaded to the expected directories.

ISSUE ID: LOGS-002
SEVERITY: High
CATEGORY: Integration failures
FILES: d:\AI\Jarvis\logs\app.log, d:\AI\Jarvis\logs\audit.jsonl.bak
DESCRIPTION: The application repeatedly fails to connect to Ollama and cloud LLM providers, leading to timeout errors and planner degradation.
ROOT CAUSE: The Ollama service is either not running locally, or the network configuration is preventing access. Additionally, cloud fallbacks are left unconfigured.
EVIDENCE: `2026-02-21 20:26:14,068 WARNING jarvis.controller: Ollama not reachable - planner will degrade gracefully` and `All cloud providers failed unconfigured. LLM or are`.
POTENTIAL IMPACT: The core agent loop and planning capabilities cannot function properly without an LLM, rendering the AI assistant mostly inoperable.
RECOMMENDED FIX: Ensure Ollama is running and accessible at the configured URL/port. Also, configure valid cloud fallback credentials (e.g., API keys).

ISSUE ID: LOGS-003
SEVERITY: Medium
CATEGORY: File handling issues
FILES: d:\AI\Jarvis\logs\audit.jsonl.corrupted
DESCRIPTION: The audit log file is marked as corrupted, indicating a failure to properly write or close the JSONL audit file.
ROOT CAUSE: Likely an improper application shutdown, an unexpected crash, or a race condition during file writing that caused the JSONL structure to become malformed or truncated.
EVIDENCE: The presence of the file `d:\AI\Jarvis\logs\audit.jsonl.corrupted` alongside `audit.jsonl.bak`.
POTENTIAL IMPACT: Loss of critical audit trails or failures in the dashboard/log viewer when attempting to parse the audit logs.
RECOMMENDED FIX: Implement safe file writing (e.g., atomic writes or write-ahead logging) and ensure graceful shutdown handlers accurately flush and close the audit file to prevent corruption.
