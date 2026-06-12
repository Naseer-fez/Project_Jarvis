# Acceptance Criteria

The rebuilt system is considered successful if and only if:

1. **Dependency Inversion Maintained**: `grep -r "import integrations" core/` returns zero results.
2. **HitL Enforcement**: The system halts and requests user input before executing any tool decorated with `RiskLevel.HIGH`.
3. **Loop Resilience**: A simulated network drop during an external API call from `integrations` does not crash the `AgentLoopEngine`; the error must be gracefully logged as a failed `ToolObservation`.
4. **Memory Retention**: The system can recall a simulated conversation fact across a daemon reboot (verifying `sqlite` and `chromadb` persistence).
5. **Full Test Pass**: The `tests/` directory must execute with 100% pass rate over unit and integration suites using mocked tool routers.