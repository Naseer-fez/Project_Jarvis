# File Report: loader.py
## Role: Dependency Analyst

### 1. Library Requirements
- `importlib` (Standard Library)
- `inspect` (Standard Library)
- `logging` (Standard Library)
- `pathlib` (Standard Library)
- `typing` (Standard Library)
- `integrations.base` (Local): imports `BaseIntegration`

### 2. Service Dependencies
- None. Loads files from the local filesystem (`clients/*.py`).

### 3. Hidden Execution Links
- Dynamically imports all python modules in `integrations/clients/` that do not start with `_`.
- Inspects all classes in the imported modules for subclasses of `BaseIntegration`.
- Instantiates these classes without any arguments (`cls()`), expecting them to have parameterless initializers or optional args.
- If `instance.is_available()` is truthy, registers it with the provided `registry`.

### 4. Assumptions & API Contracts
- Assumes the `clients` directory exists alongside `loader.py`.
- Ignores subclasses not defined in the module being loaded (`cls.__module__ != module_name`).
- Assumes the passed `registry` parameter has a `.register()` method.
- Fails safely on import/init/availability errors and logs them instead of crashing.

### 5. Configuration Variables
- `config` parameter is accepted in `load_all` for backward compatibility but is unused (ignores it and mentions "integrations use env-only config").

### 6. Prompts Found
- None.
