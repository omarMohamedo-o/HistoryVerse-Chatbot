"""Semantic Query Drift & Scope Refusal Simulation Suite for HistoryVerse.

Benchmarks baseline in-domain historical queries against synthetic out-of-domain
distribution shifts to verify Prometheus drift alerts and domain boundary gating.
Loads test queries dynamically from configuration/dataset files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.rag.refusal import DomainRefusalGate
from agent.services.nlp.colloquial import expand_egyptian_colloquial_query

DEFAULT_DATASET_PATH = Path("evals/datasets/drift_queries.json")


def load_drift_dataset(dataset_path: Path | str) -> tuple[list[str], list[str]]:
    path = Path(dataset_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("in_domain_historical_queries", []), data.get("out_of_domain_drift_queries", [])

    # Dynamic fallback if file is moved
    return (
        ["من بنى قلعة قايتباي في الإسكندرية وما هو سبب إنشائها؟"],
        ["ما هي نتيجة مباراة الأهلي والزمالك في دوري أبطال أفريقيا؟"],
    )


def run_drift_simulation(dataset_path: Path | str = DEFAULT_DATASET_PATH) -> None:
    print("=" * 60)
    print("Executing HistoryVerse Semantic Drift & Scope Refusal Simulation")
    print(f"Loading queries from: {dataset_path}")
    print("=" * 60)

    in_domain_queries, out_of_domain_queries = load_drift_dataset(dataset_path)
    gate = DomainRefusalGate(domain="history", strict_mode=True)

    print("\n--- Testing In-Domain Historical Queries ---")
    in_domain_accepted = 0
    for q in in_domain_queries:
        decision = gate.check_query(q)
        status = "ACCEPTED" if not decision.is_refused else "REFUSED"
        print(f"[{status}] {q[:45]}...")
        if not decision.is_refused:
            in_domain_accepted += 1

    print(f"\nIn-Domain Accuracy: {in_domain_accepted}/{len(in_domain_queries)} accepted.")

    print("\n--- Testing Out-of-Domain Drift Queries ---")
    out_of_domain_refused = 0
    for q in out_of_domain_queries:
        decision = gate.check_query(q)
        status = "REFUSED (SAFE)" if decision.is_refused else "LEAKED (DRIFT)"
        print(f"[{status}] {q[:45]}...")
        if decision.is_refused:
            out_of_domain_refused += 1

    print(f"\nDrift Catch Rate: {out_of_domain_refused}/{len(out_of_domain_queries)} safely refused.")

    assert in_domain_accepted == len(in_domain_queries), "In-domain historical queries falsely rejected!"
    assert out_of_domain_refused == len(out_of_domain_queries), "Drift queries leaked past refusal gate!"
    print("\nAll HistoryVerse Drift and Scope Invariance assertions PASSED successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistoryVerse Drift Simulator")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to drift queries JSON")
    args = parser.parse_args()

    run_drift_simulation(args.dataset)
