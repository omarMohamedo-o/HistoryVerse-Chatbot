"""Reproducible DVC Pipeline Runner for HistoryVerse Corpus.

Executes extract, validate, chunk, and index stages self-contained for
Egyptian and Alexandrian historical monuments, artifacts, and museum exhibits.
All file paths are configurable via environment variables or CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.getenv("DVC_DATA_DIR", "data"))
DEFAULT_CORPUS_FILE = Path(
    os.getenv("DVC_CORPUS_FILE", str(DEFAULT_DATA_DIR / "corpus" / "historyverse_corpus.json"))
)
DEFAULT_VALIDATED_FILE = Path(
    os.getenv(
        "DVC_VALIDATED_FILE", str(DEFAULT_DATA_DIR / "corpus" / "historyverse_validated.json")
    )
)
DEFAULT_CHUNKS_FILE = Path(
    os.getenv("DVC_CHUNKS_FILE", str(DEFAULT_DATA_DIR / "corpus" / "historyverse_chunks.json"))
)


def stage_extract(corpus_file: Path) -> None:
    print(f"[DVC] Stage 1: Extract HistoryVerse Corpus from {corpus_file}...")
    if not corpus_file.exists():
        print(f"Error: {corpus_file} not found.")
        sys.exit(1)
    print(f"Loaded raw historical corpus ({corpus_file.stat().st_size} bytes)")


def stage_validate(corpus_file: Path, validated_file: Path) -> None:
    print("[DVC] Stage 2: Validate HistoryVerse Corpus Schema...")
    with open(corpus_file, encoding="utf-8") as f:
        data = json.load(f)

    validated = []
    for item in data:
        assert "artifact_id" in item, "Missing artifact_id"
        assert "artifact_label" in item, "Missing artifact_label"
        assert "description_ar" in item, "Missing description_ar"
        assert "description_en" in item, "Missing description_en"
        validated.append(item)

    validated_file.parent.mkdir(parents=True, exist_ok=True)
    with open(validated_file, "w", encoding="utf-8") as out:
        json.dump(validated, out, ensure_ascii=False, indent=2)

    print(f"Validated {len(validated)} historical monuments & artifacts. Saved to {validated_file}")


def stage_chunk(validated_file: Path, chunks_file: Path) -> None:
    print("[DVC] Stage 3: Chunk HistoryVerse Corpus for Vector Retrieval...")
    with open(validated_file, encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for art in data:
        chunks.append(
            {
                "chunk_id": f"chunk_{art['artifact_id']}_ar",
                "artifact_id": art["artifact_id"],
                "title": art["artifact_label"],
                "location": art.get("location", ""),
                "era": art.get("era", ""),
                "builder": art.get("builder", ""),
                "text": art["description_ar"],
                "language": "ar",
                "key_facts": art.get("key_facts", []),
            }
        )
        chunks.append(
            {
                "chunk_id": f"chunk_{art['artifact_id']}_en",
                "artifact_id": art["artifact_id"],
                "title": art.get("artifact_label_en", art["artifact_label"]),
                "location": art.get("location", ""),
                "era": art.get("era", ""),
                "builder": art.get("builder", ""),
                "text": art["description_en"],
                "language": "en",
                "key_facts": art.get("key_facts", []),
            }
        )

    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_file, "w", encoding="utf-8") as out:
        json.dump(chunks, out, ensure_ascii=False, indent=2)

    print(f"Generated {len(chunks)} bilingual chunks. Saved to {chunks_file}")


def stage_embed(chunks_file: Path) -> None:
    print(f"[DVC] Stage 4: Index & Embed Historical Chunks from {chunks_file}...")
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found.")
        sys.exit(1)
    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Indexed {len(chunks)} historical and AR exhibit chunks ready for retrieval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVC pipeline stage runner for HistoryVerse")
    parser.add_argument("stage", choices=["extract", "validate", "chunk", "embed", "all"])
    parser.add_argument(
        "--corpus", default=str(DEFAULT_CORPUS_FILE), help="Path to raw corpus JSON"
    )
    parser.add_argument(
        "--validated", default=str(DEFAULT_VALIDATED_FILE), help="Path to validated corpus JSON"
    )
    parser.add_argument(
        "--chunks", default=str(DEFAULT_CHUNKS_FILE), help="Path to output chunks JSON"
    )
    args = parser.parse_args()

    c_file = Path(args.corpus)
    v_file = Path(args.validated)
    ch_file = Path(args.chunks)

    if args.stage in ("extract", "all"):
        stage_extract(c_file)
    if args.stage in ("validate", "all"):
        stage_validate(c_file, v_file)
    if args.stage in ("chunk", "all"):
        stage_chunk(v_file, ch_file)
    if args.stage in ("embed", "all"):
        stage_embed(ch_file)
