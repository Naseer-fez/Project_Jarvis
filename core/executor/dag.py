"""
core/executor/dag.py
────────────────────
Dependency parsing and topological sorting for DAG execution plans.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


class DependencyGraphError(ValueError):
    """Raised when there is an issue with the dependency graph (e.g. cycle)."""


class PlanDAG:
    def __init__(self, steps: List[Dict[str, Any]]):
        self.steps = steps
        # Map step ID (normalized to string) -> step definition
        self.step_map: Dict[str, Dict[str, Any]] = {str(step.get("id", "")): step for step in steps if step.get("id")}
        self.adj_list: Dict[str, Set[str]] = {str(step_id): set() for step_id in self.step_map}
        self.in_degree: Dict[str, int] = {str(step_id): 0 for step_id in self.step_map}

        # Build graph: parent -> child edge. A parent must complete before child executes.
        for step_id, step in self.step_map.items():
            depends_on = step.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]

            for dep in depends_on:
                dep_str = str(dep)
                # Ensure the dependency exists in our steps
                if dep_str in self.step_map:
                    self.adj_list[dep_str].add(step_id)
                    self.in_degree[step_id] += 1
                else:
                    # Ignore missing dependencies to prevent graph breakage
                    pass

    def topological_sort(self) -> List[str]:
        """Perform Kahn's algorithm for topological sorting and cycle detection."""
        in_deg = self.in_degree.copy()
        queue = [node for node, deg in in_deg.items() if deg == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for neighbor in self.adj_list[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(self.step_map):
            raise DependencyGraphError("Circular dependency detected in execution plan.")

        return sorted_nodes
