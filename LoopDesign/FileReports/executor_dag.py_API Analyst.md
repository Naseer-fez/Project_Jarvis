# API Analyst Report: executor\dag.py

## Dependencies
- `from __future__ import annotations`
- `from typing import Any`
- `from typing import Dict`
- `from typing import List`
- `from typing import Set`

## Schemas & API Contracts (Classes)

### Class `DependencyGraphError(ValueError)`
> Raised when there is an issue with the dependency graph (e.g. cycle).



### Class `PlanDAG`
**Methods:**
- `def __init__(self, steps: List[Dict[str, Any]])`
- `def topological_sort(self) -> List[str]`
  - *Perform Kahn's algorithm for topological sorting and cycle detection.*

