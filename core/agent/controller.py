"""
core/agent/controller.py — DEPRECATED SHIM.

MainController was the original V1 controller. It has been superseded by
JarvisControllerV2 in core/controller_v2.py, which is the canonical entry
point for all new code.

This file now re-exports JarvisControllerV2 as MainController for any legacy
references that haven't been updated yet. It will be removed in a future
release. Do not add new functionality here.

Migration path:
    Old (DEPRECATED):
        from core.agent.controller import MainController
        ctrl = MainController(config=cfg)

    New (canonical):
        from core.controller_v2 import JarvisControllerV2
        ctrl = JarvisControllerV2(config=cfg)
"""

import logging
import warnings

from core.controller_v2 import JarvisControllerV2

warnings.warn(
    "core.agent.controller.MainController is deprecated and will be removed. "
    "Use core.controller_v2.JarvisControllerV2 instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)
logger.warning(
    "DEPRECATED: core.agent.controller imported. Use core.controller_v2 instead."
)


# Backward-compatible shim alias
MainController = JarvisControllerV2

__all__ = ["MainController"]
