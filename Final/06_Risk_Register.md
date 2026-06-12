# Risk Register

## Severity High
1. **Silent Asynchronous Failures**: Unhandled exceptions in background tasks spawned by `Controller` might not crash the loop but leave the system in an inconsistent state.
2. **Dynamic Resolution Fragility**: Refactoring class paths without updating configurations or string constants will break startup.

## Severity Medium
1. **Graceful Shutdown Timeouts**: Hardcoded timeout for `controller.shutdown()`. If IO or DB operations stall, data corruption or partial writes can occur.
2. **Dashboard Binding Conflicts**: Hardcoded fallback ports can collide if multiple instances or zombie processes hold the port.
3. **Audit Logger Panic**: The `_safe_audit` call assumes the logging subsystem is robust. A filesystem failure here might cause silent loss of session metrics or crash the teardown phase.

## Severity Low
1. **Type Safety Gaps**: Statically analyzing the dynamically loaded configuration and plugins is limited (MyPy might miss these).
2. **Environment Variable Fallbacks**: Missing `.env` can default to unstable `development` mode settings in production paths.
