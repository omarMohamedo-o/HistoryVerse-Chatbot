"""Unit and benchmark tests for the AI Evals Engine."""

import pytest
from fastapi.testclient import TestClient

from historyverse.services.evaluation.ai_evals import AIEvalsEngine, EvalSample
from tests.conftest import MockLLMProvider


def test_ai_eval_metrics_calculation():
    engine = AIEvalsEngine(MockLLMProvider())

    # 1. Context Relevance
    q = "What was the Pharos Lighthouse?"
    ctx = ["The Lighthouse of Alexandria, or Pharos of Alexandria, guided ships."]
    ctx_rel = engine.evaluate_context_relevance(q, ctx)
    assert ctx_rel >= 0.5

    # 2. Faithfulness
    grounded_ans = "The Lighthouse of Alexandria guided ships into the ancient harbor."
    faith = engine.evaluate_faithfulness(grounded_ans, ctx)
    assert faith >= 0.7

    # 3. Answer Relevance
    ans_rel = engine.evaluate_answer_relevance(q, grounded_ans)
    assert ans_rel >= 0.6

    # 4. Factual Accuracy against Golden Ideal
    ideal = "The Lighthouse was on the island of Pharos in Alexandria to guide ships."
    fact_acc = engine.evaluate_factual_accuracy(grounded_ans, ideal)
    assert fact_acc >= 0.5

    # 5. Unlearning Compliance
    clean_score, _ = engine.evaluate_unlearning_compliance(
        "I have no records regarding this retracted claim.", ["alien pyramid"]
    )
    assert clean_score == 1.0

    leak_score, msg = engine.evaluate_unlearning_compliance(
        "The alien pyramid was confirmed underwater.", ["alien pyramid"]
    )
    assert leak_score == 0.0
    assert "Violation" in msg


@pytest.mark.asyncio
async def test_ai_evals_sample_and_benchmark():
    engine = AIEvalsEngine(MockLLMProvider())

    sample = EvalSample(
        id="test-1",
        category="factuality",
        question="What was the Serapeum?",
        contexts=["The Serapeum was an ancient temple in Alexandria dedicated to Serapis."],
        ideal="The Serapeum of Alexandria was a classical temple for Serapis.",
    )

    result = await engine.evaluate_sample(sample)
    assert result.passed is True
    assert result.composite_score >= 0.70

    # Run over benchmark dataset
    report = await engine.run_benchmark("evals/datasets/historical_evals.jsonl")
    assert report.total_samples == 5
    assert report.passed_samples >= 4
    assert report.pass_rate >= 0.80
    assert report.grade in ["A+", "A"]

    # Verify markdown generation
    md = engine.generate_markdown_report(report)
    assert "# AI Evaluation Benchmark Summary" in md
    assert "Faithfulness / Groundedness" in md
    assert "| `eval-001` |" in md


def test_evaluate_benchmark_api_route(client: TestClient):
    response = client.post("/v1/evaluate/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "total_samples" in data
    assert "pass_rate" in data
    assert "grade" in data
    assert len(data["results"]) == 5
