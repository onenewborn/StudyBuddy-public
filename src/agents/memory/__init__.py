"""
Memory module public API.
"""
from src.agents.memory.profile_schema import (
    ChatMessage,
    ChatSession,
    ChatTurn,
    MemoryContext,
    PerformanceRecord,
    StudyStreak,
    UserMemory,
    UserPreferences,
)
from src.agents.memory.memory_service import (
    COMPRESSION_THRESHOLD,
    MAX_CONTEXT_TURNS,
    MemoryService,
)
from src.agents.memory.memory_agent import ChatCompressor

__all__ = [
    "UserPreferences",
    "StudyStreak",
    "PerformanceRecord",
    "ChatMessage",
    "ChatTurn",
    "ChatSession",
    "UserMemory",
    "MemoryContext",
    "MemoryService",
    "COMPRESSION_THRESHOLD",
    "MAX_CONTEXT_TURNS",
    "ChatCompressor",
]
