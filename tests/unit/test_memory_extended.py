"""Unit tests for conversation memory manager with Redis backend and agent memory sync methods."""

import json
import sys
from unittest.mock import MagicMock

from agent.config.settings import Settings
from agent.services.evaluation.hallucination import HallucinationDetector
from agent.services.memory.agent_memory import AgentMemory, AgentMemoryManager
from agent.services.memory.manager import ConversationMemory as ConversationManager


def test_conversation_manager_with_mock_redis(monkeypatch):
    """Verify ConversationMemory routes operations to Redis backend when redis is connected."""
    mock_redis = MagicMock()
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_redis.Redis.from_url.return_value = mock_client
    monkeypatch.setitem(sys.modules, "redis", mock_redis)

    # Mock Redis responses
    turn_json = json.dumps([{"role": "user", "content": "Hello Redis"}])
    mock_client.get.side_effect = lambda k: None if k == "historyverse:session:empty_key" else turn_json
    mock_client.keys.return_value = ["historyverse:session:conv1", "historyverse:session:empty_key"]
    mock_client.scan_iter.return_value = ["historyverse:session:conv1", "historyverse:session:empty_key"]

    settings = Settings(redis_url="redis://localhost:6379/0")
    manager = ConversationManager(settings=settings)
    assert manager._redis_client is not None

    # Test add_exchange
    manager.add_exchange("conv1", "Hello Redis", "Greetings")
    mock_client.set.assert_called_once()

    # Test get_history
    history = manager.get_history("conv1")
    assert len(history) == 1
    assert history[0]["content"] == "Hello Redis"

    # Test clear
    manager.clear("conv1")
    mock_client.delete.assert_called_with("historyverse:session:conv1")

    # Test purge_concept
    purged = manager.purge_concept(["Hello"])
    assert purged == 1

    # Test purge_concept empty and whitespace-only
    assert manager.purge_concept([]) == 0
    assert manager.purge_concept(["   "]) == 0

    # Test Redis trimming when length > max_history * 2
    manager._max_history = 1
    mock_client.get.return_value = json.dumps([
        {"role": "user", "content": "1"}, {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"}, {"role": "assistant", "content": "4"},
    ])
    manager.add_exchange("conv1", "new_q", "new_a")

    # Test Redis delete error during clear
    mock_client.delete.side_effect = RuntimeError("Redis down")
    manager.clear("conv1")

    # Test Redis purge error
    mock_client.scan_iter.side_effect = RuntimeError("Scan error")
    manager.purge_concept(["test"])


def test_conversation_manager_redis_import_error(monkeypatch):
    """Verify ConversationMemory falls back to local storage when redis is not importable."""
    monkeypatch.setitem(sys.modules, "redis", None)
    settings = Settings(redis_url="redis://localhost:6379/0")
    manager = ConversationManager(settings=settings)
    assert manager._redis_client is None


def test_agent_memory_sync_methods_and_clear():
    """Verify AgentMemory and AgentMemoryManager synchronous purge and clear methods."""
    mock_backend = MagicMock()
    mock_backend.purge_concept.return_value = 5

    mem = AgentMemory(agent_id="agent-archaeologist", backend=mock_backend)
    mem.clear("session-01")
    mock_backend.clear.assert_called_once_with("agent-archaeologist::session-01")

    mgr = AgentMemoryManager(shared_backend=mock_backend)
    all_purged = mgr.purge_from_all_agents(["forbidden_secret"])
    assert all_purged == 5
    mock_backend.purge_concept.assert_called_with(["forbidden_secret"])

    agent_purged = mgr.purge_from_agent("agent-archaeologist", ["forbidden_secret"])
    assert agent_purged == 5


def test_hallucination_detector_edge_cases():
    """Verify HallucinationDetector handles empty response and short-word responses."""
    detector = HallucinationDetector()
    # Empty response -> 1.0
    assert detector.score("What was Memphis?", "") == 1.0
    assert detector.score("What was Memphis?", "   ") == 1.0

    # Response with only 1-3 letter words -> 0.5
    assert detector.score("What was Memphis?", "it is a no", source_documents=["Memphis was capital."]) == 0.5
