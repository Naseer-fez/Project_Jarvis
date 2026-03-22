"""Compatibility shim for legacy integration imports.

Older code imported integration contracts from ``integrations.base_integration``.
The canonical implementation now lives in ``integrations.base``.
"""

from __future__ import annotations

from integrations.base import BaseIntegration, IntegrationResult, RiskLevel, ToolResult

__all__ = ["BaseIntegration", "IntegrationResult", "ToolResult", "RiskLevel"]
