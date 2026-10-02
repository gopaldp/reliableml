# Copilot instructions

## Project

ReliableML is a master's thesis research prototype. It compares a baseline
ML pipeline with a proposed MLOps pipeline that adds data-quality gates
(Pandera), MLflow tracking and registry, Evidently drift monitoring, and a
FastAPI prediction service.

Stack: Python ≥ 3.11 (package `src/reliableml`, installed with
`pip install -e .`), LightGBM, scikit-learn, MLflow, Pandera, Evidently,
FastAPI. Dependencies are in `requirements.txt` and `pyproject.toml`.

How it runs (see `Makefile`; each target has a plain Python equivalent in
`scripts/`):

1. `make setup`: venv, dependencies, editable install.
2. `make data` → `scripts/generate_data.py --scenario all`
   (`make data-public` → `scripts/download_public_datasets.py`).
3. `make baseline` → `scripts/run_baseline.py`.
4. `make mlflow` (MLflow server on port 5000), then `make proposed` →
   `scripts/run_proposed_pipeline.py`.
5. `make serve` → `uvicorn reliableml.service.app:app` on port 8000.
6. Drift and production simulation: `scripts/simulate_production.py`,
   `scripts/run_drift_monitoring.py`; experiments: `scripts/run_experiments.py`;
   reproducibility: `scripts/reproduce_run.py`; thesis metrics:
   `scripts/export_thesis_metrics.py`.
7. `make compose-up`: Docker Compose (MLflow + API).
8. Tests: `tests/` (pytest).

Configuration: `configs/*.yaml` and `.env.example`.

## Documentation notes

- This repo is **already well documented**: `README.md`,
  `MANUAL_EXECUTION_GUIDE.md` and `docs/` (architecture,
  experiment_protocol, metrics_definition, reproducibility, thesis_mapping).
  Make **targeted corrections only**. Do not restructure these files, and do
  not create `docs/setup.md` or a new `docs/architecture.md`. Update the
  existing ones.
- Check every command in the docs against `Makefile` targets and
  `scripts/` arguments. Note that the Makefile uses `.venv/Scripts/activate`
  (Windows layout).
- Keep thesis wording (research question, baseline vs proposed). Don't
  add result numbers that aren't produced by the code.

## Ignore

- `mlruns/`, `artifacts/`, `reports/`, and the data under `data/` (only
  `.gitkeep` files are tracked).

## Conventions

- Keep changes small and focused. One concern per pull request.
- Base documentation on the actual code. Never invent features, metrics or
  commands. Mark anything uncertain with `TODO: confirm …`.
- Use UTF-8 for all text files.
