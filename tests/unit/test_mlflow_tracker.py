"""Unit tests for offline MLflow experiment tracker for datasets, chunks, and evaluations."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from evals.chunk_experiment import run_chunk_experiment
from langchain_core.documents import Document

from historyverse.config.settings import Settings
from historyverse.services.evaluation.ai_evals import (
    AIEvalBenchmarkReport,
    ConfusionMatrix,
    EvalResult,
    LossMetrics,
)
from historyverse.services.evaluation.mlflow_tracker import (
    ChunkExperimentSummary,
    DatasetProfile,
    MLflowExperimentTracker,
)


def test_mlflow_tracker_disabled_by_default():
    settings = Settings(mlflow_enabled=False)
    tracker = MLflowExperimentTracker(settings)
    assert tracker.is_active is False


def test_mlflow_tracker_missing_package_fallback():
    settings = Settings(mlflow_enabled=True)
    # If mlflow is not installed, it falls back gracefully
    tracker = MLflowExperimentTracker(settings)
    # Even if disabled or missing, operations return summaries safely
    summary = tracker.log_chunk_experiment(
        documents=[Document(page_content="Alexandria lighthouse")],
        chunks=[Document(page_content="Alexandria"), Document(page_content="lighthouse")],
        chunk_size=100,
        chunk_overlap=20,
    )
    assert isinstance(summary, ChunkExperimentSummary)
    assert summary.total_chunks == 2


def test_profile_dataset_jsonl(tmp_path: Path):
    dataset_file = tmp_path / "test_evals.jsonl"
    data = [
        {"id": "1", "category": "factuality", "question": "Who was Ptolemy I?"},
        {"id": "2", "category": "unlearning", "question": "Secret tunnel?"},
        {"id": "3", "category": "factuality", "question": "Where was the Serapeum?"},
    ]
    with open(dataset_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    profile = MLflowExperimentTracker.profile_dataset(dataset_file)
    assert isinstance(profile, DatasetProfile)
    assert profile.total_samples == 3
    assert profile.categories == {"factuality": 2, "unlearning": 1}
    assert len(profile.sha256_checksum) == 64
    assert profile.file_size_bytes > 0


def test_profile_dataset_nonexistent():
    profile = MLflowExperimentTracker.profile_dataset("nonexistent_file.jsonl")
    assert profile.total_samples == 0
    assert profile.sha256_checksum == ""


def test_log_chunk_experiment_metrics_computation():
    tracker = MLflowExperimentTracker(Settings(mlflow_enabled=False))
    docs = [Document(page_content="Hello world from Alexandria")]
    chunks = [
        Document(page_content="Hello world"),
        Document(page_content="world from"),
        Document(page_content="from Alexandria"),
    ]

    summary = tracker.log_chunk_experiment(
        documents=docs,
        chunks=chunks,
        chunk_size=50,
        chunk_overlap=10,
        duration_seconds=0.015,
        extra_params={"strategy": "recursive"},
    )

    assert summary.total_documents == 1
    assert summary.total_chunks == 3
    assert summary.chunk_size == 50
    assert summary.chunk_overlap == 10
    assert summary.mean_chunk_chars > 0
    assert summary.duration_seconds == 0.015


def test_log_chunk_experiment_with_mocked_mlflow():
    mock_mlflow = MagicMock()
    mock_context = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_context

    tracker = MLflowExperimentTracker(Settings(mlflow_enabled=False))
    tracker._mlflow = mock_mlflow

    docs = [Document(page_content="Historical text")]
    chunks = [Document(page_content="Hist"), Document(page_content="text")]

    summary = tracker.log_chunk_experiment(
        documents=docs,
        chunks=chunks,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert summary.total_chunks == 2
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    assert mock_mlflow.log_dict.call_count == 2


def test_log_eval_benchmark_with_mocked_mlflow(tmp_path: Path):
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-12345"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    tracker = MLflowExperimentTracker(Settings(mlflow_enabled=False))
    tracker._mlflow = mock_mlflow

    report = AIEvalBenchmarkReport(
        timestamp="2026-10-21T00:00:00Z",
        total_samples=1,
        passed_samples=1,
        pass_rate=1.0,
        mean_faithfulness=0.95,
        mean_answer_relevance=0.90,
        mean_context_relevance=1.0,
        mean_factual_accuracy=0.98,
        unlearning_compliance_rate=1.0,
        grade="A (Pass)",
        confusion_matrix=ConfusionMatrix(),
        loss_metrics=LossMetrics(),
        results=[
            EvalResult(
                id="eval-1",
                category="factuality",
                question="Q1",
                generated_answer="A1",
                faithfulness=0.95,
                answer_relevance=0.90,
                factual_accuracy=0.98,
                composite_score=0.94,
                passed=True,
            )
        ],
    )

    run_id = tracker.log_eval_benchmark(
        report=report,
        parameters={"model": "gemini-1.5-flash"},
        artifacts_dir=tmp_path,
    )

    assert run_id == "test-run-12345"
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    mock_mlflow.log_dict.assert_called_once()


def test_run_chunk_experiment_matrix():
    results = run_chunk_experiment(
        source_dir="./nonexistent_test_dir",
        chunk_sizes=[200, 400],
        overlaps=[50],
        mlflow_enabled=False,
    )

    assert len(results) == 2
    for r in results:
        assert r["total_chunks"] >= 1
        assert r["mean_chars"] > 0
        assert r["duration_seconds"] >= 0.0
