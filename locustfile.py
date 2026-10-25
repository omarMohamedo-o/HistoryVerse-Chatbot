"""Locust Load Testing Profile for HistoryVerse Multi-Agent Platform.

Dynamically loads museum artifacts and benchmark queries directly from the
production corpus (data/corpus/historyverse_corpus.json) to simulate real-world visitor traffic.

Usage:
    locust -f locustfile.py --headless -u 50 -r 10 --run-time 1m --host http://localhost:8000
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from locust import HttpUser, between, task

CORPUS_PATH = Path(os.getenv("HISTORYVERSE_CORPUS_PATH", "data/corpus/historyverse_corpus.json"))

# Dynamically load artifacts and build queries from corpus
def load_corpus_data() -> tuple[list[str], list[str]]:
    artifacts: list[str] = []
    queries: list[str] = []

    if CORPUS_PATH.exists():
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    label_ar = item.get("artifact_label")
                    label_en = item.get("artifact_label_en")
                    if label_ar:
                        artifacts.append(label_ar)
                        queries.append(f"من هو باني {label_ar} وما هو تاريخ إنشائها؟")
                        queries.append(f"ما هي القيمة التاريخية والأثرية لـ {label_ar}؟")
                    if label_en:
                        artifacts.append(label_en)
                        queries.append(f"What is the historical significance of {label_en}?")
        except Exception:
            pass

    if not artifacts:
        artifacts = ["قلعة قايتباي", "منارة الإسكندرية", "حجر رشيد"]
        queries = ["ما هو تاريخ قلعة قايتباي في الإسكندرية؟"]

    return artifacts, queries


MUSEUM_ARTIFACTS, HISTORICAL_QUERIES = load_corpus_data()


class HistoryVerseVisitor(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def check_health(self) -> None:
        """Health check probe hitting the gateway."""
        self.client.get("/v1/health", name="/v1/health")

    @task(2)
    def query_historical_guide(self) -> None:
        """Simulates visitor asking historical questions to Mora."""
        query = random.choice(HISTORICAL_QUERIES)
        payload = {
            "query": query,
            "agent_id": "historical_guide",
            "language": "ar" if any("\u0600" <= c <= "\u06FF" for c in query) else "en",
        }
        self.client.post("/v1/chat", json=payload, name="/v1/chat")

    @task(2)
    def explore_ar_artifact(self) -> None:
        """Simulates AR visitor querying information about a museum statue/monument."""
        artifact = random.choice(MUSEUM_ARTIFACTS)
        query = random.choice(HISTORICAL_QUERIES)
        payload = {
            "artifact_name": artifact,
            "query": query,
        }
        self.client.post("/v1/ar/chat", json=payload, name="/v1/ar/chat")

    @task(1)
    def check_metrics(self) -> None:
        """Prometheus metrics endpoint scrape."""
        self.client.get("/metrics", name="/metrics")
