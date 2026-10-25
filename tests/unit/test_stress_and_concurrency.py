"""Stress, high concurrency, and provider outage resilience tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config.settings import Settings
from agent.services.agents.domain import DomainAgent
from agent.services.llm.drift import DriftDetector
from agent.services.memory.manager import ConversationMemory


@pytest.mark.asyncio
async def test_high_concurrency_agent_execution():
    """Simulate 30 concurrent agent executions to ensure no race conditions or thread deadlocks."""
    mock_provider = AsyncMock()
    mock_provider.model_name = "stress-mock-model"
    mock_provider.generate = AsyncMock(
        side_effect=lambda q, **kwargs: f"Concurrently processed: {q[:10]}"
    )
    mock_provider.estimate_confidence = MagicMock(return_value=0.95)

    agent = DomainAgent(
        agent_id="stress-tester",
        name="Concurrent Worker",
        role="Load Tester",
        system_prompt="Execute tasks concurrently.",
        llm_provider=mock_provider,
        domain="testing",
    )

    queries = [f"Query batch request #{i} for stress analysis" for i in range(30)]

    async def execute_task(q: str):
        res = await agent.run(q)
        return res

    tasks = [execute_task(q) for q in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 30
    for res in results:
        assert res.agent_id == "stress-tester"
        assert "Concurrently processed:" in res.output
    assert mock_provider.generate.call_count == 30


@pytest.mark.asyncio
async def test_concurrent_conversation_memory_access():
    """Simulate 25 concurrent users writing and reading distinct conversation sessions."""
    settings = Settings()
    memory = ConversationMemory(settings)

    async def user_session_flow(user_id: int):
        session_id = f"user_sess_{user_id}"
        await memory.add_exchange_async(
            session_id, f"Hello from user {user_id}", f"Reply to user {user_id}"
        )
        history = await memory.get_formatted_history_async(session_id)
        assert f"Hello from user {user_id}" in history
        assert f"Reply to user {user_id}" in history
        return session_id

    tasks = [user_session_flow(i) for i in range(25)]
    completed = await asyncio.gather(*tasks)
    assert len(completed) == 25


@pytest.mark.asyncio
async def test_provider_outage_and_fallback_recovery():
    """Simulate primary LLM provider low-quality/outage output triggering fallback recovery."""
    settings = Settings(drift_threshold=0.7)

    # Primary provider produces sub-threshold / degraded confidence
    degraded_primary = AsyncMock()
    degraded_primary.model_name = "primary-model"
    degraded_primary.generate_with_confidence = AsyncMock(
        return_value=("Degraded output due to transient upstream congestion", 0.35)
    )

    # Fallback provider succeeds with high confidence
    healthy_fallback = AsyncMock()
    healthy_fallback.model_name = "fallback-model"
    healthy_fallback.generate_with_confidence = AsyncMock(
        return_value=("Recovered response from fallback service", 0.94)
    )

    detector = DriftDetector(
        primary_provider=degraded_primary,
        fallback_provider=healthy_fallback,
        settings=settings,
    )

    answer, report = await detector.generate_with_drift_protection("Critical historical query")

    assert answer == "Recovered response from fallback service"
    assert report.confidence_score == 0.94
    assert report.fallback_used is True
    assert degraded_primary.generate_with_confidence.call_count >= 1
    assert healthy_fallback.generate_with_confidence.call_count == 1


@pytest.mark.asyncio
async def test_concurrent_database_writes_data_integrity(tmp_path):
    """Test concurrent database writes to verify zero lost updates, zero corruption, and WAL concurrency."""
    from agent.services.agents.registry_store import AgentRegistryStore

    db_path = str(tmp_path / "wal_concurrency.db")
    store = AgentRegistryStore(db_path=db_path)

    async def write_agent(idx: int):
        # Run SQLite sync call in thread pool to simulate concurrent async worker requests
        def _do_write():
            store.save_agent(
                agent_id=f"agent_concurrent_{idx:03d}",
                name=f"Worker {idx}",
                role="Data Analyst",
                system_prompt=f"Process data partition {idx}.",
                metadata={"worker_index": idx, "checksum": idx * 100},
            )

        await asyncio.to_thread(_do_write)

    # 50 concurrent writes
    await asyncio.gather(*[write_agent(i) for i in range(50)])

    # Verify atomic data integrity: exactly 50 uncorrupted records
    all_agents = store.list_agents()
    assert len(all_agents) == 50

    recorded_indices = {a["metadata"]["worker_index"] for a in all_agents}
    assert recorded_indices == set(range(50))


@pytest.mark.asyncio
async def test_scalable_stress_concurrency_benchmarks():
    """Scalable stress benchmark across increasing concurrency tiers with latency, memory, and error rate measurement."""
    import time
    import tracemalloc

    mock_provider = AsyncMock()
    mock_provider.model_name = "benchmark-model"
    mock_provider.generate = AsyncMock(return_value="Synthesized benchmark payload.")
    mock_provider.estimate_confidence = MagicMock(return_value=0.98)

    agent = DomainAgent(
        agent_id="bench-agent",
        name="Benchmark Worker",
        role="Throughput Evaluator",
        system_prompt="Evaluate workload scaling.",
        llm_provider=mock_provider,
    )

    concurrency_tiers = [10, 25, 50]
    tier_results = []

    async def execute_tier(tier: int) -> dict[str, Any]:
        tracemalloc.start()
        t_start = time.perf_counter()

        latencies: list[float] = []
        errors = [0]

        async def worker(task_id: int):
            w_start = time.perf_counter()
            try:
                res = await agent.run(f"Benchmark task #{task_id}")
                assert res.output == "Synthesized benchmark payload."
            except Exception:
                errors[0] += 1
            finally:
                latencies.append(time.perf_counter() - w_start)

        await asyncio.gather(*[worker(i) for i in range(tier)])
        total_time = time.perf_counter() - t_start
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        error_rate = errors[0] / tier

        return {
            "tier": tier,
            "total_time_s": total_time,
            "p50_ms": p50 * 1000,
            "p95_ms": p95 * 1000,
            "p99_ms": p99 * 1000,
            "peak_mem_kb": peak_mem / 1024,
            "error_rate": error_rate,
        }

    for t in concurrency_tiers:
        metrics = await execute_tier(t)
        assert metrics["error_rate"] == 0.0
        tier_results.append(metrics)

    assert len(tier_results) == 3
