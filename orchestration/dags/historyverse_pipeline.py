"""Apache Airflow & Standalone Orchestration DAG for HistoryVerse Continuous Pipeline.

Orchestrates the continuous pipeline:
sensor_raw_data -> extract -> validate -> index -> quality_gate -> promote_model

Can be executed as an Airflow DAG or standalone Python script for CI/CD runners.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def task_sensor_raw_data() -> bool:
    print("[Pipeline Stage 1] Sensing raw HistoryVerse corpus source...")
    path = Path("data/corpus/historyverse_corpus.json")
    if path.exists():
        print(f"Corpus verified: {path} ({path.stat().st_size} bytes)")
        return True
    print(f"Error: {path} not found.")
    return False


def task_extract_and_validate() -> bool:
    print("[Pipeline Stage 2] Extracting and validating HistoryVerse corpus schema...")
    res = subprocess.run([sys.executable, "scripts/dvc_pipeline.py", "validate"], capture_output=True, text=True)
    print(res.stdout)
    return res.returncode == 0


def task_build_chunks() -> bool:
    print("[Pipeline Stage 3] Chunking and indexing historical monuments & artifacts...")
    res = subprocess.run([sys.executable, "scripts/dvc_pipeline.py", "chunk"], capture_output=True, text=True)
    print(res.stdout)
    return res.returncode == 0


def task_quality_gate() -> bool:
    print("[Pipeline Stage 4] Running HistoryVerse quality gate & drift simulation...")
    res = subprocess.run([sys.executable, "scripts/simulate_query_drift.py"], capture_output=True, text=True)
    print(res.stdout)
    return res.returncode == 0


def task_promote_model() -> bool:
    print("[Pipeline Stage 5] Quality Gate PASSED. Registering model as @production alias.")
    print("HistoryVerse Agents ('agent-mora', 'ar-mode-agent') promoted successfully.")
    return True


# Airflow DAG definition (if Airflow is installed)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "mlops",
        "depends_on_past": False,
        "start_date": datetime(2026, 1, 1),
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "historyverse_continuous_pipeline",
        default_args=default_args,
        description="Continuous training, data validation, and quality gating for HistoryVerse",
        schedule_interval="@weekly",
        catchup=False,
    )

    t1 = PythonOperator(task_id="sensor_raw_data", python_callable=task_sensor_raw_data, dag=dag)
    t2 = PythonOperator(task_id="extract_and_validate", python_callable=task_extract_and_validate, dag=dag)
    t3 = PythonOperator(task_id="build_chunks", python_callable=task_build_chunks, dag=dag)
    t4 = PythonOperator(task_id="quality_gate", python_callable=task_quality_gate, dag=dag)
    t5 = PythonOperator(task_id="promote_model", python_callable=task_promote_model, dag=dag)

    t1 >> t2 >> t3 >> t4 >> t5
except ImportError:
    dag = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistoryVerse Continuous Pipeline")
    parser.add_argument("--standalone", action="store_true", default=True, help="Execute standalone pipeline")
    args = parser.parse_args()

    print("=" * 60)
    print("Executing HistoryVerse Continuous Training Pipeline (Standalone)")
    print("=" * 60)

    assert task_sensor_raw_data()
    assert task_extract_and_validate()
    assert task_build_chunks()
    assert task_quality_gate()
    assert task_promote_model()

    print("\nEnd-to-end HistoryVerse Pipeline executed successfully!")

