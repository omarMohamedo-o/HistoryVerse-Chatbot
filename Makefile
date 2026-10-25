# HistoryVerse Makefile — Automated Developer and Evaluator Commands

.PHONY: help install test dvc pipeline drift regression perturbation loadtest lint format clean

help:
	@echo "Available commands for HistoryVerse:"
	@echo "  make install       Install package in editable mode"
	@echo "  make test          Run full test suite"
	@echo "  make dvc           Run reproducible 4-stage DVC pipeline"
	@echo "  make pipeline      Run continuous training orchestration pipeline"
	@echo "  make drift         Run semantic query drift simulation"
	@echo "  make regression    Run historical monument regression tests"
	@echo "  make perturbation  Run metamorphic faithfulness perturbation tests"
	@echo "  make loadtest      Run Locust concurrent user load test"
	@echo "  make lint          Check code formatting and linting rules"
	@echo "  make format        Auto-format code with ruff"
	@echo "  make run           Launch API gateway with uvicorn"

install:
	pip install -e .

test:
	python3 tests/unit/test_enhancements_suite.py -v
	python3 tests/regression/test_history_cases.py
	PYTHONPATH=src python3 tests/evals/test_faithfulness_perturbation.py

dvc:
	python3 scripts/dvc_pipeline.py all

pipeline:
	python3 orchestration/dags/historyverse_pipeline.py

drift:
	python3 scripts/simulate_query_drift.py

regression:
	python3 tests/regression/test_history_cases.py

perturbation:
	PYTHONPATH=src python3 tests/evals/test_faithfulness_perturbation.py

loadtest:
	locust -f locustfile.py --headless -u 20 -r 5 --run-time 30s --host http://localhost:8000

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn agent.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
