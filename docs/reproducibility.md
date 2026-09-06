# Reproducibility Guide

## Goal

Ensure that any researcher can reproduce exact results from this thesis project.

## Prerequisites

- Python 3.11 or 3.12
- Git (for version control)
- 8GB RAM
- 10GB disk space

## Step-by-Step Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd reliableml

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: source .venv/Scripts/activate

# Install exact dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Datasets

```bash
# Generate all scenarios with default seed (42)
python scripts/generate_data.py --scenario all

# Verify data generation
ls data/processed/
# Should contain: clean_*, data_quality_failure_*, mild_drift_*, severe_drift_*, performance_degradation_*
```

**Expected Output**:
- `clean_train.parquet`: ~7000 rows
- `clean_val.parquet`: ~1500 rows
- `clean_test.parquet`: ~1500 rows
- Similar splits for other scenarios

### 3. Run Baseline Pipeline

```bash
python scripts/run_baseline.py --scenario clean
```

**Expected Output**:
- File: `reports/baseline_run_summary.json`
- Metrics: RMSE ~40-60, R² ~0.75-0.85
- Duration: 2-5 seconds

### 4. Run Proposed Pipeline

```bash
# Start MLflow tracking server (separate terminal)
mlflow server --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./artifacts/mlruns \
  --host 127.0.0.1 --port 5000

# Run proposed pipeline
python scripts/run_proposed_pipeline.py --scenario clean
```

**Expected Output**:
- File: `reports/proposed_pipeline_summary.json`
- MLflow run logged with metrics
- Model registered in MLflow Model Registry
- Metrics: Similar to baseline (same data/model)

### 5. Simulate Production and Monitor Drift

```bash
# Simulate severe drift
python scripts/simulate_production.py --scenario severe_drift

# Run drift monitoring
python scripts/run_drift_monitoring.py --scenario severe_drift
```

**Expected Output**:
- `reports/drift_report.html`: Visual drift report
- `reports/drift_decision.json`: Decision = "RETRAIN_REQUIRED"
- Drift share > 50%

### 6. Verify Reproducibility

```bash
python scripts/reproduce_run.py --scenario clean
```

**Expected Output**:
- `reports/reproducibility_report.json`
- All checks: PASS
- Exit code: 0

### 7. Export Thesis Metrics

```bash
python scripts/export_thesis_metrics.py
```

**Expected Output**:
- `reports/comparison_metrics.csv`
- `reports/comparison_summary.md`

## Verification Checksums

### Dataset Fingerprints (Seed=42, 10000 rows)

These SHA-256 hashes should match if data generation is deterministic:

```
clean_train.parquet: <hash will vary by exact implementation>
```

**Note**: Exact hashes depend on pandas/numpy versions. Verify by running reproducibility check.

### Metric Ranges (Clean Scenario)

| Metric | Expected Range |
|--------|----------------|
| MAE | 8-15 |
| RMSE | 35-65 |
| R² | 0.70-0.90 |
| MAPE | 0.10-0.20 |

If your results fall outside these ranges, check:
1. Random seed is 42
2. Hyperparameters match `configs/base.yaml`
3. No data preprocessing errors

## Known Sources of Variation

### Acceptable Variations

1. **Floating-Point Precision**: Minor differences (< 1e-5) due to hardware
2. **Timing Measurements**: Duration varies by CPU load
3. **LightGBM Versions**: Different versions may produce slightly different trees

### Unacceptable Variations

1. **Different Metrics**: RMSE differs by > 5% → Check seed and config
2. **Failed Gates**: Quality gates should consistently PASS/FAIL per scenario
3. **Fingerprint Mismatch**: Dataset hashes differ → Data generation issue

## Debugging Reproducibility Issues

### Issue: Different Metrics Across Runs

**Diagnosis**:
```bash
# Check seed in config
grep random_seed configs/base.yaml

# Check data fingerprint
python -c "
from reliableml.data.fingerprint import compute_fingerprint
import pandas as pd
df = pd.read_parquet('data/processed/clean_train.parquet')
print(compute_fingerprint(df))
"
```

**Solution**: Ensure seed is fixed and data generation uses same code path.

### Issue: Tests Failing

**Diagnosis**:
```bash
pytest tests/ -v --tb=short
```

**Common Causes**:
- Package version mismatch
- Missing dependencies
- Data files not generated

**Solution**:
```bash
pip install -r requirements.txt --force-reinstall
python scripts/generate_data.py --scenario clean --rows 500
pytest tests/test_data_generator.py -v
```

### Issue: MLflow Database Locked

**Diagnosis**:
```
sqlite3.OperationalError: database is locked
```

**Solution**:
```bash
# Stop MLflow server
pkill -f "mlflow server"

# Remove lock files
rm -f mlruns.db-shm mlruns.db-wal

# Restart server
mlflow server --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./artifacts/mlruns \
  --host 127.0.0.1 --port 5000
```

## Docker Reproduction

```bash
# Build and run all services
docker compose up --build

# Services available:
# - MLflow: http://localhost:5000
# - API: http://localhost:8000/docs

# Run tests inside container
docker compose exec prediction-service pytest tests/ -v
```

## Continuous Integration

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
1. Lint check (ruff)
2. Test suite (pytest)
3. Smoke test (data generation + baseline pipeline)

## Publishing Results

For thesis publication:

1. **Archive Code**: Tag release with thesis submission date
2. **Archive Data**: Save generated datasets to persistent storage
3. **Document Environment**: Include `pip freeze > requirements.lock.txt`
4. **Save Reports**: Commit `reports/*.json` and `reports/*.md` to git

## Contact for Reproducibility Issues

If you cannot reproduce results:
1. Check this guide carefully
2. Verify Python and package versions
3. Review error messages in detail
4. Create GitHub issue with environment details

## Reproducibility Checklist

- [ ] Python 3.11 or 3.12 installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from `requirements.txt`
- [ ] Data generated with `--scenario all`
- [ ] Baseline pipeline runs successfully
- [ ] Proposed pipeline runs successfully
- [ ] MLflow UI accessible at localhost:5000
- [ ] Drift monitoring produces reports
- [ ] Reproducibility check exits with code 0
- [ ] All tests pass (`pytest tests/`)
- [ ] Metrics within expected ranges
