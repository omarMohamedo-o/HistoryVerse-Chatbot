"""End-to-End Integration Verification Suite for HistoryVerse AI.

Validates the full enterprise lifecycle across:
1. Multi-Agent Orchestration & Strict Memory Isolation
2. Model Context Protocol (MCP) & Tool Execution Harness
3. Selective Machine Unlearning with Adversarial Probing
4. Cryptographic Audit Certificate Minting
5. AI Evaluation Benchmark Suite (RAG Triad, Confusion Matrix, Calibration Loss)
"""

from unittest.mock import MagicMock

import pytest

from historyverse.config.settings import Settings
from historyverse.core.enums import UnlearningStatus
from historyverse.core.schemas import UnlearningRequest
from historyverse.services.agents.historical import HistoryVerseAgent
from historyverse.services.agents.researcher import ResearcherAgent
from historyverse.services.evaluation.ai_evals import AIEvalsEngine
from historyverse.services.mcp.harness import AgentHarness
from historyverse.services.mcp.tools import (
    ArtifactInspectionTool,
    HistoricalSearchTool,
    ToolRegistry,
)
from historyverse.services.memory.agent_memory import AgentMemoryManager
from historyverse.services.rag.retriever import ChromaRetriever
from historyverse.services.unlearning.engine import UnlearningEngine
from historyverse.services.unlearning.registry import TombstoneRegistry
from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_end_to_end_complete_platform_lifecycle():
    settings = Settings(agent_persona_name="Mora")
    session_id = "e2e-session-gold"

    # Step 1: Multi-Agent Orchestration & Isolated Memory
    memory_manager = AgentMemoryManager(settings=settings)
    mem_mora = memory_manager.get_agent_memory("historyverse-mora")
    mem_researcher = memory_manager.get_agent_memory("archival-researcher")

    llm_mora = MockLLMProvider("The Great Library of Alexandria was founded under Ptolemy I.")
    llm_researcher = MockLLMProvider("Archival papyri cataloged at Oxford reference Oxyrhynchus.")

    mora_agent = HistoryVerseAgent(llm_mora, memory=mem_mora)
    researcher_agent = ResearcherAgent(llm_researcher, memory=mem_researcher)

    # Mora turn
    resp_mora = await mora_agent.run(
        "Tell me about the Library of Alexandria",
        context={"context": "Historical founding of Alexandria.", "session_id": session_id},
    )
    assert "Ptolemaic dynasty" in resp_mora.output

    # Researcher turn
    resp_researcher = await researcher_agent.run(
        "Tell me about the Serapeum",
        context={"context": "Oxford Papyrus collection.", "session_id": session_id},
    )
    assert "Serapeum" in resp_researcher.output

    # Verify strict memory isolation between agents
    mora_turns = await mem_mora.get_history_async(session_id)
    researcher_turns = await mem_researcher.get_history_async(session_id)

    assert len(mora_turns) == 2
    assert "Ptolemaic dynasty" in mora_turns[1]["content"]
    assert "Serapeum" not in mora_turns[1]["content"]

    assert len(researcher_turns) == 2
    assert "Serapeum" in researcher_turns[1]["content"]
    assert "Ptolemaic dynasty" not in researcher_turns[1]["content"]

    # Step 2: Model Context Protocol (MCP) & Defensive Agent Harness
    registry = ToolRegistry()
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Alexander the Great laid out the city grid of Alexandria in 331 BC."
    mock_retriever.retrieve.return_value = [mock_doc]

    registry.register(HistoricalSearchTool(retriever=mock_retriever))
    registry.register(ArtifactInspectionTool())

    harness = AgentHarness(registry=registry)

    tool_call_res = await harness.call_tool(
        "search_historical_archives",
        {"query": "Alexander city grid", "top_k": 1},
    )
    assert tool_call_res.is_success is True
    assert "Alexander the Great" in str(tool_call_res.data)
    assert tool_call_res.duration_ms >= 0.0

    # Test harness execution with agent
    harness_agent_res = await harness.execute_agent_with_harness(
        mora_agent,
        query="Who planned the street layout of Alexandria?",
        max_turns=2,
    )
    assert harness_agent_res.agent_id == "historyverse-mora"
    assert harness_agent_res.output is not None

    # Step 3: Selective Machine Unlearning & Adversarial Validation
    # Add a contaminated turn to Mora's memory to simulate a debunked myth
    await mem_mora.add_exchange_async(
        session_id,
        "Is there a secret subterranean labyrinth under Alexandria?",
        "Yes, rumors claim a secret subterranean labyrinth exists.",
    )
    assert len(await mem_mora.get_history_async(session_id)) == 4

    tombstones = TombstoneRegistry()
    mock_vector_retriever = MagicMock(spec=ChromaRetriever)
    mock_vector_retriever.delete_documents_by_keywords.return_value = 1
    mock_vector_retriever.retrieve.return_value = []

    unlearning_llm = MockLLMProvider("I have no verified historical evidence of any labyrinth.")
    unlearning_engine = UnlearningEngine(
        llm_provider=unlearning_llm,
        memory_manager=memory_manager,
        retriever=mock_vector_retriever,
        tombstone_registry=tombstones,
    )

    # Execute forgetting request
    unlearn_req = UnlearningRequest(
        target_concept="Secret Subterranean Labyrinth Myth",
        reason="Archaeological consensus debunked claims of secret subterranean labyrinth.",
        forbidden_keywords=["subterranean labyrinth", "secret labyrinth"],
        purge_memory=True,
        purge_rag=True,
    )
    unlearn_report = await unlearning_engine.unlearn(unlearn_req)

    assert unlearn_report.validation_passed is True
    assert unlearn_report.status == UnlearningStatus.FORGOTTEN_VERIFIED
    assert unlearn_report.leakage_rate == 0.0
    assert len(unlearn_report.probes) >= 3
    assert unlearn_report.certificate_id.startswith("CERT-UNLEARN-")

    # Verify contaminated turn was purged from Mora's memory while preserving valid turn
    cleaned_mora_turns = await mem_mora.get_history_async(session_id)
    assert len(cleaned_mora_turns) == 2
    assert "Ptolemaic dynasty" in cleaned_mora_turns[1]["content"]
    assert "labyrinth" not in str(cleaned_mora_turns)

    # Step 4: AI Evaluation Benchmark Suite (RAG Triad & Loss Calibration)
    eval_llm = MockLLMProvider(
        "Ptolemy I Soter established the Library of Alexandria as part of the Mouseion."
    )
    eval_engine = AIEvalsEngine(eval_llm)

    benchmark_summary = await eval_engine.run_benchmark("evals/datasets/historical_evals.jsonl")

    assert benchmark_summary.grade in ["A+", "A"]
    assert benchmark_summary.total_samples == 5
    assert benchmark_summary.passed_samples >= 4
    assert benchmark_summary.pass_rate >= 0.80

    # Validate confusion matrix
    cm = benchmark_summary.confusion_matrix
    assert cm.true_positives >= 1
    assert cm.false_positives == 0
    assert cm.accuracy == 1.0
    assert cm.precision == 1.0
    assert cm.f1_score == 1.0

    # Validate loss and calibration metrics
    assert benchmark_summary.loss_metrics.cross_entropy_loss >= 0.0
    assert benchmark_summary.loss_metrics.brier_score >= 0.0
    assert benchmark_summary.loss_metrics.perplexity >= 1.0
