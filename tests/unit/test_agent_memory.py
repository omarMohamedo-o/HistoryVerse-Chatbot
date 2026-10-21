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


async def test_agent_memory_async_isolation_and_unlearning():
    """Verify non-blocking async operations maintain strict agent isolation and unlearning."""
    shared_backend = ConversationMemory(Settings())
    mem_a = AgentMemory("agent-async-a", backend=shared_backend)
    mem_b = AgentMemory("agent-async-b", backend=shared_backend)
    session_id = "session-async-1"

    await mem_a.add_exchange_async(
        session_id, "Who was Cleopatra?", "Cleopatra was Queen of Egypt."
    )
    await mem_b.add_exchange_async(
        session_id, "Who was Julius Caesar?", "Caesar was Roman dictator."
    )

    history_a = await mem_a.get_history_async(session_id)
    history_b = await mem_b.get_history_async(session_id)

    assert len(history_a) == 2
    assert "Cleopatra" in history_a[0]["content"]
    assert len(history_b) == 2
    assert "Caesar" in history_b[0]["content"]

    formatted_a = await mem_a.get_formatted_history_async(session_id)
    assert "Cleopatra" in formatted_a

    # Unlearn only for agent A
    purged = await mem_a.unlearn_concept_async(["Cleopatra"], session_id=session_id)
    assert purged == 2
    assert len(await mem_a.get_history_async(session_id)) == 0

    # Agent B must be untouched
    assert len(await mem_b.get_history_async(session_id)) == 2

    await mem_b.clear_async(session_id)
    assert len(await mem_b.get_history_async(session_id)) == 0


async def test_agent_memory_manager_async_purges():
    manager = AgentMemoryManager()
    mem_1 = manager.get_agent_memory("agent-x")
    mem_2 = manager.get_agent_memory("agent-y")

    await mem_1.add_exchange_async("sess-1", "Query Sphinx", "Sphinx in Giza")
    await mem_2.add_exchange_async("sess-2", "Query Nile", "Nile River flooding")

    purged_single = await manager.purge_from_agent_async("agent-x", ["Sphinx"])
    assert purged_single == 2

    purged_all = await manager.purge_from_all_agents_async(["Nile"])
    assert purged_all == 2
