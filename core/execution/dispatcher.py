"""
core/execution/dispatcher.py
"""

import logging
from typing import Any, Dict

from core.executor.engine import DAGExecutor

logger = logging.getLogger("Jarvis.Execution.Dispatcher")

class DispatchPipeline:
    """
    High-level wrapper around DAGExecutor.
    Enforces a hardcoded max_recursion_depth to prevent unbounded execution loops.
    """
    
    def __init__(self, executor: DAGExecutor):
        self.executor = executor
        self.max_recursion_depth = 5
        
    async def execute(self, plan: Dict[str, Any], context: Any, current_depth: int = 0) -> Dict[str, Any]:
        """
        Executes a plan via DAGExecutor, checking recursion depth.
        """
        if current_depth > self.max_recursion_depth:
            err_msg = f"DispatchPipeline max recursion depth of {self.max_recursion_depth} breached."
            logger.critical(err_msg)
            raise RecursionError(err_msg)
            
        return await self.executor.execute(plan, context)
