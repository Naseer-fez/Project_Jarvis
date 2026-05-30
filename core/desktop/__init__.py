"""Closed-loop desktop reliability primitives."""

from core.desktop.actions import DesktopActionExecutor
from core.desktop.contracts import (
    ApprovalDecision,
    DesktopAction,
    DesktopActionResult,
    DesktopActionStatus,
    DesktopActionType,
    DesktopChange,
    DesktopObservation,
    DesktopRiskTier,
    ScreenTarget,
)
from core.desktop.mission import (
    DesktopMissionExecutor,
    DesktopMissionStatus,
    MissionExecutionRecord,
    MissionStepRecord,
    RecoveryDecision,
)
from core.desktop.observation import DesktopObserver

__all__ = [
    "ApprovalDecision",
    "DesktopAction",
    "DesktopActionExecutor",
    "DesktopActionResult",
    "DesktopActionStatus",
    "DesktopActionType",
    "DesktopChange",
    "DesktopMissionExecutor",
    "DesktopMissionStatus",
    "DesktopObservation",
    "DesktopObserver",
    "DesktopRiskTier",
    "MissionExecutionRecord",
    "MissionStepRecord",
    "RecoveryDecision",
    "ScreenTarget",
]
