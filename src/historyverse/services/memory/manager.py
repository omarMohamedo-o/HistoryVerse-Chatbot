"""Thread-safe multi-session conversation memory manager."""

from __future__ import annotations

import logging
import threading
from typing import Any

from historyverse.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["ConversationMemory"]


class ConversationMemory:
    """Manages scoped conversation history per session with maximum buffer limits."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._max_history = self._settings.max_conversation_history
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def add_exchange(self, conversation_id: str, question: str, answer: str) -> None:
        """Append a question-answer turn to the session history."""
        with self._lock:
            if conversation_id not in self._sessions:
                self._sessions[conversation_id] = []

            session = self._sessions[conversation_id]
            session.append({"role": "user", "content": question})
            session.append({"role": "assistant", "content": answer})

            if len(session) > self._max_history * 2:
                self._sessions[conversation_id] = session[-self._max_history * 2 :]

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return raw history entries for a given session."""
        with self._lock:
            return list(self._sessions.get(conversation_id, []))

    def get_formatted_history(self, conversation_id: str) -> str:
        """Format session exchanges into string for prompt context injection."""
        with self._lock:
            entries = self._sessions.get(conversation_id, [])
            if not entries:
                return "No prior conversation."

            lines: list[str] = []
            for entry in entries:
                prefix = "User" if entry["role"] == "user" else "Mora"
                lines.append(f"{prefix}: {entry['content']}")
            return "\n".join(lines)

    def clear(self, conversation_id: str) -> None:
        """Reset conversation history for a given session."""
        with self._lock:
            if conversation_id in self._sessions:
                del self._sessions[conversation_id]
