"""Unit tests for MLflowTracker covering full lifecycle, dataset profiling, and exception recovery."""

import sys
from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.config.settings import Settings
from agent.services.evaluation.ai_evals import (
    AIEvalBenchmarkReport,
    ConfusionMatrix,
    LossMetrics,
)
from agent.services.evaluation.mlflow_tracker import (
    MLflowExperimentTracker as MLflowTracker,
)


def test_mlflow_initialization_success(monkeypatch):
    """Verify MLflowTracker successfully initializes when mlflow library is present."""
    mock_mlflow = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    settings = Settings(mlflow_enabled=True, mlflow_tracking_uri="http://localhost:5000", mlflow_experiment_name="test-exp")
    tracker = MLflowTracker(settings=settings)
    assert tracker.is_active is True
    mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    mock_mlflow.set_experiment.assert_called_once_with("test-exp")


def test_mlflow_initialization_exception(monkeypatch):
    """Verify MLflowTracker falls back to offline mode when set_tracking_uri raises an exception."""
    mock_mlflow = MagicMock()
    mock_mlflow.set_tracking_uri.side_effect = ConnectionRefusedError("Tracking server unreachable")
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    settings = Settings(mlflow_enabled=True)
    tracker = MLflowTracker(settings=settings)
    assert tracker.is_active is False


def test_dataset_profiling_jsonl_empty_and_corrupt_lines(tmp_path):
    """Verify dataset profiler handles empty lines and malformed JSON lines in jsonl files."""
    jsonl_file = tmp_path / "dataset.jsonl"
    jsonl_file.write_text(
        '{"question": "Q1", "category": "pharaohs"}\n\n'
        'invalid json line here\n'
        '{"question": "Q2", "category": "pyramids"}\n'
    )

    settings = Settings(mlflow_enabled=False)
    tracker = MLflowTracker(settings=settings)
    profile = tracker.profile_dataset(jsonl_file)
    assert profile.total_samples == 2
    assert profile.categories.get("pharaohs") == 1
    assert profile.categories.get("pyramids") == 1


def test_dataset_profiling_json_corrupt(tmp_path):
    """Verify dataset profiler gracefully handles invalid JSON in .json files."""
    json_file = tmp_path / "dataset.json"
    json_file.write_text("CORRUPTED_JSON_CONTENT{[")

    settings = Settings(mlflow_enabled=False)
    tracker = MLflowTracker(settings=settings)
    profile = tracker.profile_dataset(json_file)
    assert profile.total_samples == 0


def test_log_ingestion_chunks_exception_handled(monkeypatch):
    """Verify MLflow chunk logging exceptions do not raise or break ingest summary."""
    mock_mlflow = MagicMock()
    mock_mlflow.log_dict.side_effect = RuntimeError("Artifact upload error")
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    settings = Settings(mlflow_enabled=True)
    tracker = MLflowTracker(settings=settings)
    chunks = [Document(page_content="Text chunk 1", metadata={"source": "test.txt"})]

    summary = tracker.log_chunk_experiment(
        documents=[Document(page_content="Full text")],
        chunks=chunks,
        chunk_size=500,
        chunk_overlap=50,
    )
    assert summary.total_chunks == 1


def test_log_eval_benchmark_with_dataset_and_failure(monkeypatch, tmp_path):
    """Verify benchmark run logging profiles dataset and handles run failures safely."""
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = RuntimeError("Failed to start MLflow run")
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    settings = Settings(mlflow_enabled=True)
    tracker = MLflowTracker(settings=settings)

    dataset_file = tmp_path / "eval_data.jsonl"
    dataset_file.write_text('{"question": "Q", "category": "history"}\n')

    report = AIEvalBenchmarkReport(
        timestamp="2026-10-22T00:00:00Z",
        total_samples=10,
        passed_samples=9,
        pass_rate=0.9,
        mean_faithfulness=0.95,
        mean_answer_relevance=0.92,
        mean_context_relevance=0.88,
        mean_factual_accuracy=0.94,
        unlearning_compliance_rate=1.0,
        grade="A+",
        confusion_matrix=ConfusionMatrix(
            true_positives=9, false_positives=0, true_negatives=1, false_negatives=0
        ),
        loss_metrics=LossMetrics(cross_entropy_loss=0.1, brier_score=0.05, perplexity=1.1),
    )

    run_id = tracker.log_eval_benchmark(report=report, dataset_path=dataset_file)
    # When start_run raises, logger logs warning and run_id remains None
    assert run_id is None
