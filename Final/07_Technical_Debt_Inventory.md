# Technical Debt Inventory

## 1. Module Coupling
- The `core.runtime` heavily relies on dynamic reflection (`_load_controller_class`) which hides the static dependency graph from linters (mypy, ruff).
- **Debt Payload**: High. Refactoring the controller class or path breaks runtime silently until executed.

## 2. Incomplete Type Hinting
- `core/runtime/entrypoint.py` relies on `typing.Any` for the `controller` and `shutdown` objects, meaning no static checks for methods like `run_cli()` or `wait()`.
- **Debt Payload**: Medium. Missed refactorings can lead to runtime `AttributeError`.

## 3. Global Exception Suppression
- `contextlib.suppress(Exception)` is used during shutdown logic to hide teardown failures.
- **Debt Payload**: Medium. Masked errors make memory/resource leak tracking difficult.

## 4. Hardcoded Environment Variables
- `JARVIS_ENV` implicitly drives behavior without centralized validation schemas (e.g., Pydantic).
- **Debt Payload**: Low. Configuration sprawl is manageable but currently unstructured.
