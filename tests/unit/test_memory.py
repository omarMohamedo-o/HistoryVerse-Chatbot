"""Unit tests for ConversationMemory manager."""

from unittest.mock import MagicMock
import json

from historyverse.config.settings import Settings
from historyverse.services.memory.manager import ConversationMemory


def test_conversation_memory_history():
    settings = Settings(max_conversation_history=3)
    memory = ConversationMemory(settings)

    cid = "user-123"
    memory.add_exchange(cid, "Hello", "Greetings! I am Mora.")
    memory.add_exchange(cid, "Tell me about Serapis", "Serapis is a Greco-Egyptian deity.")

    history = memory.get_history(cid)
    assert len(history) == 4

    formatted = memory.get_formatted_history(cid)
    assert "User: Hello" in formatted
    assert "Mora: Greetings!" in formatted


def test_conversation_memory_trim():
    settings = Settings(max_conversation_history=2)
    memory = ConversationMemory(settings)

    cid = "user-trim"
    for i in range(5):
        memory.add_exchange(cid, f"Q{i}", f"A{i}")

    history = memory.get_history(cid)
    assert len(history) <= 4


def test_conversation_memory_clear():
    memory = ConversationMemory()
    memory.add_exchange("session-1", "Q", "A")
    assert len(memory.get_history("session-1")) == 2

    memory.clear("session-1")
    assert len(memory.get_history("session-1")) == 0


def test_conversation_memory_redis_backend():
    settings = Settings(redis_url="redis://fake:6379/0")
    memory = ConversationMemory(settings)

    mock_client = MagicMock()
    mock_store: dict[str, str] = {}

    def mock_get(key):
        return mock_store.get(key)

    def mock_set(key, value, ex=None):
        mock_store[key] = value

    def mock_del(key):
        mock_store.pop(key, None)

    mock_client.get.side_effect = mock_get
    mock_client.set.side_effect = mock_set
    mock_client.delete.side_effect = mock_del

    memory._redis_client = mock_client

    cid = "redis-session-1"
    memory.add_exchange(cid, "Who founded Alexandria?", "Alexander the Great in 331 BC.")

    history = memory.get_history(cid)
    assert len(history) == 2
    assert history[0]["content"] == "Who founded Alexandria?"

    formatted = memory.get_formatted_history(cid)
    assert "Alexander the Great" in formatted

    memory.clear(cid)
    assert memory.get_history(cid) == []


def test_conversation_memory_redis_failure_fallback():
    settings = Settings(redis_url="redis://fake:6379/0")
    memory = ConversationMemory(settings)

    mock_client = MagicMock()
    mock_client.get.side_effect = ConnectionError("Redis server unreachable")
    mock_client.set.side_effect = ConnectionError("Redis server unreachable")
    memory._redis_client = mock_client

    cid = "fallback-session"
    memory.add_exchange(cid, "Hello", "Hi there!")

    history = memory.get_history(cid)
    assert len(history) == 2
