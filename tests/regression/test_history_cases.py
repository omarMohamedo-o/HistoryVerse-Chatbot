"""HistoryVerse Historical & Monument Regression Test Suite.

Asserts deterministic factual retrieval, builder metadata, and historical periods
across Ancient Egyptian and Alexandrian monuments in the HistoryVerse corpus.
Configurable via environment variable or default relative path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CORPUS_PATH = Path(os.getenv("HISTORYVERSE_CORPUS_PATH", "data/corpus/historyverse_corpus.json"))


def load_corpus(path: Path = DEFAULT_CORPUS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_artifact(corpus, artifact_id: str):
    for art in corpus:
        if art["artifact_id"] == artifact_id:
            return art
    return None


def test_qaitbay_citadel_monument_facts():
    corpus = load_corpus()
    art = find_artifact(corpus, "qaitbay_citadel")
    assert art is not None, "qaitbay_citadel not found in corpus"
    assert "قايتباي" in art["artifact_label"]
    assert "1477" in art["year_built"]
    assert "منارة الإسكندرية" in art["description_ar"]
    assert "Mamluk" in art["era"]


def test_alexandria_lighthouse_wonder_facts():
    corpus = load_corpus()
    art = find_artifact(corpus, "alexandria_lighthouse")
    assert art is not None, "alexandria_lighthouse not found in corpus"
    assert "عجائب العالم القديم" in art["description_ar"]
    assert "بطليموس" in art["description_ar"]
    assert "فاروس" in art["artifact_label"]


def test_rosetta_stone_decipherment_facts():
    corpus = load_corpus()
    art = find_artifact(corpus, "rosetta_stone")
    assert art is not None, "rosetta_stone not found in corpus"
    assert "شامبليون" in art["description_ar"]
    assert "الهيروغليفية" in art["description_ar"]
    assert "196" in art["year_built"]


def test_alexandria_library_scholarship_facts():
    corpus = load_corpus()
    art = find_artifact(corpus, "alexandria_library")
    assert art is not None, "alexandria_library not found in corpus"
    assert "بطليموس الأول" in art["description_ar"]
    assert "الموسيون" in art["description_ar"]


def test_giza_pyramids_khufu_facts():
    corpus = load_corpus()
    art = find_artifact(corpus, "giza_pyramids")
    assert art is not None, "giza_pyramids not found in corpus"
    assert "خوفو" in art["description_ar"]
    assert "أبو الهول" in art["description_ar"]


if __name__ == "__main__":
    test_qaitbay_citadel_monument_facts()
    test_alexandria_lighthouse_wonder_facts()
    test_rosetta_stone_decipherment_facts()
    test_alexandria_library_scholarship_facts()
    test_giza_pyramids_khufu_facts()
    print("All 5 HistoryVerse Regression Tests PASSED successfully!")
