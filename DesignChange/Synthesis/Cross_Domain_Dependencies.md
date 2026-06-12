# Cross-Domain Dependencies

A topological map of inter-module dependency across domains, excluding standard library and external PyPI packages.

## Global Dependency Flow
`root` → `dashboard` → `integrations` → `core` → `audit`

- **`audit`**: Lowest level. Has no internal dependencies. Relies only on standard libraries.
- **`core`**: The monolithic core. Depends directly on `audit`. It does **not** import `integrations` or `dashboard` directly to avoid circular dependencies (dependency inversion is used via registries).
- **`integrations`**: Depends heavily on `core.types`, `core.capability.base`, and `core.registry.registry` to bind its tools.
- **`dashboard`**: Depends on `core` for the controller implementations, health checks, and state management.
- **`tests`**: Depends on all domains for extensive unit and integration testing.

## Dependency Inversion implementation
`core.registry.CapabilityRegistry` acts as the dependency inversion point. While `core` needs to execute tools defined in `integrations`, it does not import them. Instead, `integrations.loader.IntegrationLoader` (invoked at startup by `root`) registers the available clients into the registry, allowing `core.executor.engine` to invoke them dynamically by name.