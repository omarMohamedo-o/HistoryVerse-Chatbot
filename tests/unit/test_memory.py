"""Unit tests for ConversationMemory manager."""

from unittest.mock import MagicMock

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


async def test_conversation_memory_async_non_blocking_methods():
    settings = Settings(max_conversation_history=4)
    memory = ConversationMemory(settings)
    cid = "async-session-1"

    await memory.add_exchange_async(cid, "Tell me about Giza", "The Pyramids are in Giza.")
    history = await memory.get_history_async(cid)
    assert len(history) == 2
    assert history[0]["content"] == "Tell me about Giza"

    formatted = await memory.get_formatted_history_async(cid)
    assert "The Pyramids are in Giza" in formatted

    purged = await memory.purge_concept_async(["Giza"])
    assert purged == 2
    assert len(await memory.get_history_async(cid)) == 0

    await memory.add_exchange_async(cid, "New query", "New answer")
    await memory.clear_async(cid)
    assert len(await memory.get_history_async(cid)) == 0


def test_conversation_memory_scan_iter_purge():
    settings = Settings(redis_url="redis://fake:6379/0")
    memory = ConversationMemory(settings)

    mock_client = MagicMock()
    # Mock scan_iter to return matching keys
    mock_client.scan_iter.return_value = ["historyverse:session:test-scan-1"]
    mock_client.get.return_value = '[{"role": "user", "content": "Tell me about Tutankhamun"}]'

    memory._redis_client = mock_client
    purged = memory.purge_concept(["Tutankhamun"])

    assert purged == 1
    mock_client.scan_iter.assert_called_once_with(match="historyverse:session:*", count=100)
    mock_client.delete.assert_called_once_with("historyverse:session:test-scan-1")
