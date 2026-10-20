"""Unit tests for ConversationMemory manager."""

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
