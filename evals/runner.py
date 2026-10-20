"""Command-line runner for AI Evaluation benchmark suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running evals/runner.py as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from historyverse.config.settings import get_settings  # noqa: E402
from historyverse.services.evaluation.ai_evals import AIEvalsEngine  # noqa: E402
from historyverse.services.llm.factory import LLMFactory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evals.runner")


async def main() -> None:
    parser = argparse.ArgumentParser(description="HistoryVerse AI Evaluation Benchmark Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/historical_evals.jsonl",
        help="Path to evaluation dataset JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/reports",
        help="Directory to save eval summary reports",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run benchmark using deterministic mock LLM for offline testing or CI",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override LLM provider (e.g. gemini, openai, local, ollama)",
    )
    args = parser.parse_args()

    if args.mock:
        from tests.conftest import MockLLMProvider

        llm = MockLLMProvider()
    else:
        from historyverse.core.enums import LLMProvider

        settings = get_settings()
        if args.provider:
            settings.llm_provider = LLMProvider(args.provider)
        llm = LLMFactory.create(settings)

    engine = AIEvalsEngine(llm)

    logger.info("Starting AI evaluation benchmark using model: %s", llm.model_name)
    logger.info("Loading evaluation dataset from: %s", args.dataset)

    report = await engine.run_benchmark(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = Path(args.output_dir) / "eval_report.json"
    md_path = Path(args.output_dir) / "eval_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    md_content = engine.generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + md_content + "\n")
    logger.info("Benchmark complete! Reports saved to %s and %s", json_path, md_path)


if __name__ == "__main__":
    asyncio.run(main())
