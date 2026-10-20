"""Thread-safe multi-session conversation memory manager with Redis and in-memory fallback."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from historyverse.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["ConversationMemory"]


class ConversationMemory:
    """Manages scoped conversation history per session with Redis backing and in-memory fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._max_history = self._settings.max_conversation_history
        self._redis_url = self._settings.redis_url
        self._redis_ttl = self._settings.redis_ttl_seconds
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()
        self._redis_client: Any = None

        if self._redis_url:
            self._init_redis()

    def _init_redis(self) -> None:
        """Attempt to initialize connection to Redis backend."""
        try:
            import redis

            client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            client.ping()
            self._redis_client = client
            logger.info("Connected to Redis memory backend: %s", self._redis_url)
        except ImportError:
            logger.warning("redis library not installed; falling back to in-memory conversation storage.")
            self._redis_client = None
        except Exception as exc:
            logger.warning("Failed to connect to Redis (%s); falling back to in-memory storage.", exc)
            self._redis_client = None

    def _get_redis_key(self, conversation_id: str) -> str:
        return f"historyverse:session:{conversation_id}"

    def add_exchange(self, conversation_id: str, question: str, answer: str) -> None:
        """Append a question-answer turn to the session history."""
        if self._redis_client is not None:
            try:
                key = self._get_redis_key(conversation_id)
                current_raw = self._redis_client.get(key)
                session: list[dict[str, str]] = json.loads(current_raw) if current_raw else []
                session.append({"role": "user", "content": question})
                session.append({"role": "assistant", "content": answer})

                if len(session) > self._max_history * 2:
                    session = session[-self._max_history * 2 :]

                self._redis_client.set(key, json.dumps(session), ex=self._redis_ttl)
                return
            except Exception as exc:
                logger.warning("Redis write failed (%s); persisting turn in local in-memory fallback.", exc)

        with self._lock:
            if conversation_id not in self._sessions:
                self._sessions[conversation_id] = []

            session_local = self._sessions[conversation_id]
            session_local.append({"role": "user", "content": question})
            session_local.append({"role": "assistant", "content": answer})

            if len(session_local) > self._max_history * 2:
                self._sessions[conversation_id] = session_local[-self._max_history * 2 :]

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return raw history entries for a given session."""
        if self._redis_client is not None:
            try:
                key = self._get_redis_key(conversation_id)
                data = self._redis_client.get(key)
                if data:
                    return json.loads(data)
                return []
            except Exception as exc:
                logger.warning("Redis read failed (%s); retrieving from in-memory fallback.", exc)

        with self._lock:
            return list(self._sessions.get(conversation_id, []))

    def get_formatted_history(self, conversation_id: str) -> str:
        """Format session exchanges into string for prompt context injection."""
        entries = self.get_history(conversation_id)
        if not entries:
            return "No prior conversation."

        lines: list[str] = []
        for entry in entries:
            prefix = "User" if entry["role"] == "user" else "Mora"
            lines.append(f"{prefix}: {entry['content']}")
        return "\n".join(lines)

    def clear(self, conversation_id: str) -> None:
        """Reset conversation history for a given session."""
        if self._redis_client is not None:
            try:
                key = self._get_redis_key(conversation_id)
                self._redis_client.delete(key)
            except Exception as exc:
                logger.warning("Redis delete failed (%s); clearing from in-memory fallback.", exc)

        with self._lock:
            if conversation_id in self._sessions:
                del self._sessions[conversation_id]
