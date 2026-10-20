"""Unit tests verifying agent memory isolation and memory-level unlearning."""

from historyverse.config.settings import Settings
from historyverse.services.memory.agent_memory import AgentMemory, AgentMemoryManager
from historyverse.services.memory.manager import ConversationMemory


def test_agent_memory_strict_isolation():
    """Verify that turns added to Agent A do not leak into Agent B."""
    shared_backend = ConversationMemory(Settings())

    mem_mora = AgentMemory("historyverse-mora", backend=shared_backend)
    mem_researcher = AgentMemory("archival-researcher", backend=shared_backend)

    session_id = "user-session-42"

    mem_mora.add_exchange(
        session_id,
        "What was the Serapeum?",
        "The Serapeum was a temple dedicated to Serapis in Alexandria.",
    )

    # Mora should have 1 exchange (2 messages)
    mora_history = mem_mora.get_history(session_id)
    assert len(mora_history) == 2
    assert "Serapeum" in mora_history[1]["content"]

    # Researcher memory for the SAME session_id must be completely empty (isolated)
    researcher_history = mem_researcher.get_history(session_id)
    assert len(researcher_history) == 0

    # Add exchange to researcher
    mem_researcher.add_exchange(
        session_id,
        "Find archival papyri",
        "Papyri Oxyrhynchus 2200 references classical library records.",
    )
    assert len(mem_researcher.get_history(session_id)) == 2
    assert len(mem_mora.get_history(session_id)) == 2


def test_agent_memory_unlearning_purges_only_forbidden_turns():
    """Verify selective unlearning removes forbidden facts while preserving innocent turns."""
    backend = ConversationMemory(Settings())
    agent_mem = AgentMemory("historyverse-mora", backend=backend)
    session_id = "unlearn-session-1"

    agent_mem.add_exchange(
        session_id,
        "Tell me about the Library of Alexandria",
        "The Great Library was an ancient center of learning.",
    )
    agent_mem.add_exchange(
        session_id,
        "What about the secret Atlantis tunnel?",
        "Archaeologists found a secret Atlantis tunnel under the harbor.",
    )

    # 4 messages total
    assert len(agent_mem.get_history(session_id)) == 4

    # Purge the discredited Atlantis claim
    purged_count = agent_mem.unlearn_concept(["Atlantis tunnel", "secret Atlantis"])
    assert purged_count > 0

    # Remaining history should only contain the legitimate Library turns
    cleaned_history = agent_mem.get_history(session_id)
    assert len(cleaned_history) == 2
    assert "Library" in cleaned_history[0]["content"]
    assert "Atlantis" not in cleaned_history[1]["content"]


def test_agent_memory_manager_multi_agent():
    """Verify AgentMemoryManager correctly spawns and retrieves isolated agent instances."""
    manager = AgentMemoryManager()
    mem1 = manager.get_agent_memory("agent-alpha")
    mem2 = manager.get_agent_memory("agent-beta")
    mem1_again = manager.get_agent_memory("agent-alpha")

    assert mem1 is mem1_again
    assert mem1.agent_id != mem2.agent_id
