"""Thread-safe multi-session conversation memory manager with Redis and in-memory fallback."""

from __future__ import annotations

import asyncio
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
            logger.warning(
                "redis library not installed; falling back to in-memory conversation storage."
            )
            self._redis_client = None
        except Exception as exc:
            logger.warning(
                "Failed to connect to Redis (%s); falling back to in-memory storage.", exc
            )
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
                logger.warning(
                    "Redis write failed (%s); persisting turn in local in-memory fallback.", exc
                )

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

    def get_formatted_history(self, conversation_id: str, assistant_name: str | None = None) -> str:
        """Format session exchanges into string for prompt context injection."""
        entries = self.get_history(conversation_id)
        if not entries:
            return "No prior conversation."

        resolved_name = assistant_name or self._settings.agent_persona_name
        lines: list[str] = []
        for entry in entries:
            prefix = "User" if entry["role"] == "user" else resolved_name
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

    def purge_concept(self, keywords: list[str], session_prefix: str | None = None) -> int:
        """Purge any stored message containing forbidden keywords.

        If session_prefix is provided, only sessions matching the prefix are purged.
        Returns the total number of dialogue turns removed.
        """
        if not keywords:
            return 0

        lower_keywords = [k.lower().strip() for k in keywords if k.strip()]
        if not lower_keywords:
            return 0

        purged_count = 0

        if self._redis_client is not None:
            try:
                pattern = (
                    f"historyverse:session:{session_prefix}*"
                    if session_prefix
                    else "historyverse:session:*"
                )
                # Non-blocking cursor scanning instead of blocking KEYS command
                keys = (
                    list(self._redis_client.scan_iter(match=pattern, count=100))
                    if hasattr(self._redis_client, "scan_iter")
                    else self._redis_client.keys(pattern)
                )
                for key in keys:
                    raw = self._redis_client.get(key)
                    if not raw:
                        continue
                    turns: list[dict[str, str]] = json.loads(raw)
                    cleaned_turns = [
                        turn
                        for turn in turns
                        if not any(kw in turn.get("content", "").lower() for kw in lower_keywords)
                    ]
                    diff = len(turns) - len(cleaned_turns)
                    purged_count += diff
                    if diff > 0:
                        (
                            self._redis_client.set(
                                key, json.dumps(cleaned_turns), ex=self._redis_ttl
                            )
                            if cleaned_turns
                            else self._redis_client.delete(key)
                        )
            except Exception as exc:
                logger.warning(
                    "Redis purge failed (%s); proceeding to in-memory fallback purge.", exc
                )

        with self._lock:
            target_convs = [
                cid
                for cid in self._sessions
                if session_prefix is None or cid.startswith(session_prefix)
            ]
            for cid in target_convs:
                turns = self._sessions[cid]
                cleaned_turns = [
                    turn
                    for turn in turns
                    if not any(kw in turn.get("content", "").lower() for kw in lower_keywords)
                ]
                diff = len(turns) - len(cleaned_turns)
                purged_count += diff
                if diff > 0:
                    self._sessions[cid] = cleaned_turns if cleaned_turns else []
                    if not cleaned_turns:
                        del self._sessions[cid]

        logger.info("Purged %d memory turns matching keywords: %s", purged_count, lower_keywords)
        return purged_count

    # Non-blocking concurrency wrappers for async event loop pipelines
    async def add_exchange_async(self, conversation_id: str, question: str, answer: str) -> None:
        """Non-blocking turn append offloaded to threadpool."""
        await asyncio.to_thread(self.add_exchange, conversation_id, question, answer)

    async def get_history_async(self, conversation_id: str) -> list[dict[str, str]]:
        """Non-blocking history fetch offloaded to threadpool."""
        return await asyncio.to_thread(self.get_history, conversation_id)

    async def get_formatted_history_async(
        self, conversation_id: str, assistant_name: str | None = None
    ) -> str:
        """Non-blocking formatted history retrieval."""
        return await asyncio.to_thread(self.get_formatted_history, conversation_id, assistant_name)

    async def clear_async(self, conversation_id: str) -> None:
        """Non-blocking session reset offloaded to threadpool."""
        await asyncio.to_thread(self.clear, conversation_id)

    async def purge_concept_async(
        self, keywords: list[str], session_prefix: str | None = None
    ) -> int:
        """Non-blocking concept purge offloaded to threadpool."""
        return await asyncio.to_thread(self.purge_concept, keywords, session_prefix)
