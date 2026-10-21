"""Offline document chunking and text-splitting experiment runner.

Evaluates chunk size, overlap, and token length distribution across historical texts
and logs experiment results to MLflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from agentverse.config.settings import get_settings  # noqa: E402
from agentverse.services.evaluation.mlflow_tracker import (  # noqa: E402
    MLflowExperimentTracker,
)
from agentverse.services.rag.ingest import DocumentIngestor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evals.chunk_experiment")


def run_chunk_experiment(
    source_dir: str,
    chunk_sizes: list[int],
    overlaps: list[int],
    mlflow_enabled: bool = False,
    experiment_name: str | None = None,
) -> list[dict[str, int | float]]:
    """Run chunk experiments across parameter combinations."""
    settings = get_settings()
    if mlflow_enabled:
        settings.mlflow_enabled = True

    tracker = MLflowExperimentTracker(
        settings=settings,
        experiment_name=experiment_name or "agentverse-chunking-experiments",
    )

    ingestor = DocumentIngestor(settings)
    documents = ingestor.load_documents(source_dir)

    if not documents:
        logger.warning("No documents loaded from %s. Using sample fallback corpus.", source_dir)
        documents = [
            Document(
                page_content=(
                    "The Library of Alexandria in Egypt was one of the largest and most significant "
                    "libraries of the ancient world. It flourished under the patronage of the Ptolemaic "
                    "dynasty and functioned as a major center of scholarship from its construction in the "
                    "3rd century BC until the Roman conquest of Egypt."
                ),
                metadata={"source": "sample_alexandria.txt"},
            )
        ]

    results: list[dict[str, int | float]] = []

    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", "؛", "؟", "!", ".", "،", " ", ""],
            )

            start = time.perf_counter()
            chunks = splitter.split_documents(documents)
            duration = time.perf_counter() - start

            summary = tracker.log_chunk_experiment(
                documents=documents,
                chunks=chunks,
                chunk_size=size,
                chunk_overlap=overlap,
                duration_seconds=duration,
                extra_params={"source_dir": source_dir},
            )

            record: dict[str, int | float] = {
                "chunk_size": summary.chunk_size,
                "chunk_overlap": summary.chunk_overlap,
                "total_chunks": summary.total_chunks,
                "mean_chars": summary.mean_chunk_chars,
                "min_chars": summary.min_chunk_chars,
                "max_chars": summary.max_chunk_chars,
                "duration_seconds": summary.duration_seconds,
            }
            results.append(record)

            logger.info(
                "Chunking [size=%d, overlap=%d]: created %d chunks (avg %.1f chars) in %.3fs",
                size,
                overlap,
                summary.total_chunks,
                summary.mean_chunk_chars,
                duration,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HistoryVerse RAG Chunking Hyperparameter Experimenter"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="./dataset",
        help="Source directory containing documents to chunk",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[250, 500, 1000],
        help="List of chunk sizes to evaluate",
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=int,
        default=[50, 100, 200],
        help="List of chunk overlaps to evaluate",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log experiment runs to MLflow",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom MLflow experiment name",
    )
    args = parser.parse_args()

    results = run_chunk_experiment(
        source_dir=args.source_dir,
        chunk_sizes=args.chunk_sizes,
        overlaps=args.overlaps,
        mlflow_enabled=args.mlflow,
        experiment_name=args.experiment_name,
    )

    print("\n--- Chunking Hyperparameter Experiment Summary ---")
    print(
        f"{'Chunk Size':<12} | {'Overlap':<10} | {'Total Chunks':<14} | {'Mean Chars':<12} | {'Duration (s)':<12}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['chunk_size']:<12} | {r['chunk_overlap']:<10} | {r['total_chunks']:<14} | {r['mean_chars']:<12.1f} | {r['duration_seconds']:<12.4f}"
        )


if __name__ == "__main__":
    main()
