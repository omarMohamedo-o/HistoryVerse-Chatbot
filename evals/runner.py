"""Command-line runner for AI Evaluation benchmark suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from historyverse.config.settings import get_settings
from historyverse.services.evaluation.ai_evals import AIEvalsEngine
from historyverse.services.llm.factory import LLMFactory

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
    args = parser.parse_args()

    settings = get_settings()
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
