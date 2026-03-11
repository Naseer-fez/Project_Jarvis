"""Legacy entrypoints for conversation and user preference memory."""

from __future__ import annotations

from .conversation import ConversationMemory
from .preferences import PreferenceStore


class ConversationStore(ConversationMemory):
    pass


class UserMemory(PreferenceStore):
    pass


__all__ = ["ConversationStore", "UserMemory"]
