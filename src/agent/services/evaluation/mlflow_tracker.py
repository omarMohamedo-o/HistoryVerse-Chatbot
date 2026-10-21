"""Offline MLflow experimentation tracker for RAG data, chunks, and evaluation benchmarks.

Decouples offline model and retrieval hyperparameter experiments from production serving.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.config.settings import Settings, get_settings
from agent.services.evaluation.ai_evals import AIEvalBenchmarkReport

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkExperimentSummary",
    "DatasetProfile",
    "MLflowExperimentTracker",
]


@dataclass(frozen=True)
class ChunkExperimentSummary:
    """Summary metrics of a document chunking experiment."""

    total_documents: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    mean_chunk_chars: float
    min_chunk_chars: int
    max_chunk_chars: int
    duration_seconds: float


@dataclass(frozen=True)
class DatasetProfile:
    """Profile of an offline evaluation or ingestion dataset."""

    file_path: str
    total_samples: int
    file_size_bytes: int
    sha256_checksum: str
    categories: dict[str, int]


class MLflowExperimentTracker:
    """Offline experiment tracker integrating with MLflow or graceful local fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._tracking_uri = tracking_uri or self._settings.mlflow_tracking_uri
        self._experiment_name = experiment_name or self._settings.mlflow_experiment_name
        self._enabled = self._settings.mlflow_enabled
        self._mlflow: Any | None = None

        if self._enabled:
            self._init_mlflow()

    def _init_mlflow(self) -> bool:
        """Lazily initialize MLflow tracking client if available."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._mlflow = mlflow
            logger.info(
                "MLflow tracking initialized: URI=%s, Experiment=%s",
                self._tracking_uri,
                self._experiment_name,
            )
            return True
        except ImportError:
            logger.warning(
                "MLflow is enabled in settings but 'mlflow' package is not installed. "
                "Running in offline local fallback mode."
            )
            self._mlflow = None
            return False
        except Exception as exc:
            logger.warning("Failed to initialize MLflow (%s); using offline local fallback.", exc)
            self._mlflow = None
            return False

    @property
    def is_active(self) -> bool:
        """Whether live MLflow logging is active."""
        return self._mlflow is not None

    @staticmethod
    def profile_dataset(file_path: str | Path) -> DatasetProfile:
        """Extract dataset statistics, checksum, and category distributions."""
        path = Path(file_path)
        if not path.is_file():
            return DatasetProfile(
                file_path=str(path),
                total_samples=0,
                file_size_bytes=0,
                sha256_checksum="",
                categories={},
            )

        content = path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        file_size = len(content)

        categories: dict[str, int] = {}
        total_samples = 0

        if path.suffix.lower() == ".jsonl":
            for line in content.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    total_samples += 1
                    cat = obj.get("category", "default")
                    categories[cat] = categories.get(cat, 0) + 1
                except Exception:
                    pass
        elif path.suffix.lower() == ".json":
            try:
                data = json.loads(content.decode("utf-8", errors="ignore"))
                if isinstance(data, list):
                    total_samples = len(data)
                    for item in data:
                        if isinstance(item, dict):
                            cat = item.get("category", "default")
                            categories[cat] = categories.get(cat, 0) + 1
            except Exception:
                pass

        return DatasetProfile(
            file_path=str(path),
            total_samples=total_samples,
            file_size_bytes=file_size,
            sha256_checksum=sha256,
            categories=categories,
        )

    def log_chunk_experiment(
        self,
        documents: list[Any],
        chunks: list[Any],
        chunk_size: int,
        chunk_overlap: int,
        duration_seconds: float = 0.0,
        extra_params: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> ChunkExperimentSummary:
        """Log a document chunking experiment to MLflow or return summary."""
        chunk_lengths = [len(getattr(c, "page_content", str(c))) for c in chunks] if chunks else [0]
        mean_len = sum(chunk_lengths) / max(len(chunk_lengths), 1)
        min_len = min(chunk_lengths) if chunks else 0
        max_len = max(chunk_lengths) if chunks else 0

        summary = ChunkExperimentSummary(
            total_documents=len(documents),
            total_chunks=len(chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mean_chunk_chars=round(mean_len, 2),
            min_chunk_chars=min_len,
            max_chunk_chars=max_len,
            duration_seconds=round(duration_seconds, 3),
        )

        params: dict[str, Any] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_documents": len(documents),
            **(extra_params or {}),
        }
        metrics: dict[str, float] = {
            "total_chunks": float(len(chunks)),
            "mean_chunk_chars": summary.mean_chunk_chars,
            "min_chunk_chars": float(summary.min_chunk_chars),
            "max_chunk_chars": float(summary.max_chunk_chars),
            "duration_seconds": summary.duration_seconds,
        }

        if self._mlflow is not None:
            try:
                resolved_run_name = run_name or f"chunk-sz{chunk_size}-ov{chunk_overlap}"
                with self._mlflow.start_run(run_name=resolved_run_name):
                    self._mlflow.log_params(params)
                    self._mlflow.log_metrics(metrics)

                    sample_chunks = [
                        {
                            "index": idx,
                            "length": len(getattr(c, "page_content", str(c))),
                            "snippet": getattr(c, "page_content", str(c))[:200],
                        }
                        for idx, c in enumerate(chunks[:10])
                    ]
                    self._mlflow.log_dict(sample_chunks, "sample_chunks.json")
                    self._mlflow.log_dict(asdict(summary), "chunk_summary.json")
            except Exception as exc:
                logger.warning("MLflow chunk logging failed (%s); proceeding offline.", exc)

        return summary

    def log_eval_benchmark(
        self,
        report: AIEvalBenchmarkReport,
        dataset_path: str | Path | None = None,
        parameters: dict[str, Any] | None = None,
        run_name: str | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> str | None:
        """Log an AI evaluation benchmark run to MLflow."""
        params: dict[str, Any] = {
            "total_samples": report.total_samples,
            "grade": report.grade,
            **(parameters or {}),
        }
        if dataset_path:
            profile = self.profile_dataset(dataset_path)
            params["dataset_path"] = profile.file_path
            params["dataset_sha256"] = profile.sha256_checksum[:12]
            params["dataset_samples"] = profile.total_samples

        metrics: dict[str, float] = {
            "pass_rate": report.pass_rate,
            "mean_faithfulness": report.mean_faithfulness,
            "mean_answer_relevance": report.mean_answer_relevance,
            "mean_context_relevance": report.mean_context_relevance,
            "mean_factual_accuracy": report.mean_factual_accuracy,
            "unlearning_compliance_rate": report.unlearning_compliance_rate,
            "accuracy": report.confusion_matrix.accuracy,
            "precision": report.confusion_matrix.precision,
            "recall": report.confusion_matrix.recall,
            "f1_score": report.confusion_matrix.f1_score,
            "cross_entropy_loss": report.loss_metrics.cross_entropy_loss,
            "brier_score": report.loss_metrics.brier_score,
            "perplexity": report.loss_metrics.perplexity,
        }

        run_id: str | None = None
        if self._mlflow is not None:
            try:
                resolved_run_name = run_name or f"eval-{report.timestamp}"
                with self._mlflow.start_run(run_name=resolved_run_name) as run:
                    run_id = getattr(run.info, "run_id", "local-run")
                    self._mlflow.log_params(params)
                    self._mlflow.log_metrics(metrics)
                    self._mlflow.log_dict(report.model_dump(), "eval_report.json")
                    if artifacts_dir:
                        art_path = Path(artifacts_dir)
                        if art_path.is_dir():
                            self._mlflow.log_artifacts(str(art_path))
                    logger.info(
                        "Successfully logged benchmark report to MLflow (run_id=%s)", run_id
                    )
            except Exception as exc:
                logger.warning("Failed to log evaluation run to MLflow (%s); logged locally.", exc)

        return run_id
