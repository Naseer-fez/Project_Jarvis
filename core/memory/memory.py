"""Legacy umbrella memory module."""

from __future__ import annotations

from .code_store import CodeStore
from .preferences import PreferenceStore
from .user_memory import ConversationStore


class Memory(PreferenceStore):
    pass


ConversationMemory = ConversationStore
CodeMemory = CodeStore

__all__ = ["CodeMemory", "ConversationMemory", "Memory"]
