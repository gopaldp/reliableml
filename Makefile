.PHONY: help setup data data-public baseline mlflow proposed serve compose-up compose-down simulate-clean simulate-mild-drift simulate-severe-drift monitor reproduce metrics experiments test lint format clean

help: ## Show this help message
	@echo "ReliableML - Master's Thesis MLOps Pipeline"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	python -m venv .venv
	. .venv/Scripts/activate && pip install --upgrade pip
	. .venv/Scripts/activate && pip install -r requirements.txt
	. .venv/Scripts/activate && pip install -e .
	@echo "✓ Environment setup complete. Activate with: source .venv/Scripts/activate"

data: ## Generate all scenario datasets
	. .venv/Scripts/activate && python scripts/generate_data.py --scenario all

data-clean: ## Generate clean scenario dataset only
	. .venv/Scripts/activate && python scripts/generate_data.py --scenario clean

data-public: ## Download and prepare UCI public thesis datasets
	. .venv/Scripts/activate && python scripts/download_public_datasets.py

baseline: ## Run baseline pipeline on clean data
	. .venv/Scripts/activate && python scripts/run_baseline.py --scenario clean

mlflow: ## Start MLflow tracking server locally
	. .venv/Scripts/activate && mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts/mlruns --host 127.0.0.1 --port 5000

proposed: ## Run proposed MLOps pipeline on clean data
	. .venv/Scripts/activate && python scripts/run_proposed_pipeline.py --scenario clean

proposed-quality-fail: ## Run proposed pipeline on data_quality_failure scenario
	. .venv/Scripts/activate && python scripts/run_proposed_pipeline.py --scenario data_quality_failure

serve: ## Start FastAPI prediction service locally
	. .venv/Scripts/activate && uvicorn reliableml.service.app:app --host 127.0.0.1 --port 8000 --reload

compose-up: ## Start all services with Docker Compose
	docker compose up --build -d
	@echo "✓ Services started. Access:"
	@echo "  - MLflow UI: http://localhost:5000"
	@echo "  - API docs: http://localhost:8000/docs"

compose-down: ## Stop Docker Compose services
	docker compose down

simulate-clean: ## Simulate clean production data
	. .venv/Scripts/activate && python scripts/simulate_production.py --scenario clean

simulate-mild-drift: ## Simulate mild drift in production
	. .venv/Scripts/activate && python scripts/simulate_production.py --scenario mild_drift

simulate-severe-drift: ## Simulate severe drift in production
	. .venv/Scripts/activate && python scripts/simulate_production.py --scenario severe_drift

monitor: ## Run drift monitoring on clean production scenario
	. .venv/Scripts/activate && python scripts/run_drift_monitoring.py --scenario clean

monitor-severe: ## Run drift monitoring on severe drift scenario
	. .venv/Scripts/activate && python scripts/run_drift_monitoring.py --scenario severe_drift

reproduce: ## Run reproducibility experiment
	. .venv/Scripts/activate && python scripts/reproduce_run.py --scenario clean

metrics: ## Export thesis comparison metrics
	. .venv/Scripts/activate && python scripts/export_thesis_metrics.py

experiments: ## Run the repeated thesis experiment matrix
	. .venv/Scripts/activate && python scripts/run_experiments.py

test: ## Run pytest test suite
	. .venv/Scripts/activate && pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	. .venv/Scripts/activate && pytest tests/ --cov=src/reliableml --cov-report=html --cov-report=term

lint: ## Run ruff linter
	. .venv/Scripts/activate && ruff check src/ tests/ scripts/

format: ## Format code with ruff
	. .venv/Scripts/activate && ruff format src/ tests/ scripts/

clean: ## Remove generated artifacts, data, and caches
	rm -rf data/processed/* data/production/* data/reference/*
	rm -rf artifacts/* reports/* prediction_logs/*
	rm -rf mlruns.db mlruns/*
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned generated artifacts and caches"

full-experiment: data baseline proposed simulate-clean monitor metrics ## Run full thesis experiment pipeline
	@echo "✓ Full experiment completed. Check reports/ for results."
