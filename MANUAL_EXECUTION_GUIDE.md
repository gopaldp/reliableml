# ReliableML - Complete Manual Setup and Execution Guide

## 📍 Quick Reference

**Project Location:** `D:\Personal Projects\reliableml`  
**Python Version:** 3.11+ (you have 3.14.7)  
**Estimated Time:** 20-30 minutes for full workflow  
**Required Terminal:** Git Bash (NOT Windows CMD/PowerShell)

---

## ⚠️ Important: Use Git Bash

This guide uses Unix-style commands (`ls`, `cat`, `grep`, etc.) that work in **Git Bash**, not Windows Command Prompt or PowerShell.

**How to open Git Bash:**
1. Press `Windows Key` and search for "Git Bash"
2. Click to open the Git Bash terminal (black window with green text)
3. You should see a prompt like: `user@DESKTOP-XXX MINGW64 ~`

**If you see `C:\>` or `PS C:\>` you're in the wrong terminal!**

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Open Git Bash and Navigate to Project

**IMPORTANT: Use Git Bash, not Windows Command Prompt (cmd.exe)**

1. Press `Windows Key` and search for **"Git Bash"**
2. Click to open Git Bash terminal
3. Navigate to the project:

```bash
cd "/d/Personal Projects/reliableml"
pwd
# Should show: /d/Personal Projects/reliableml
```

**Note:** Git Bash uses Unix-style paths with forward slashes (`/d/...`) and supports Unix commands like `ls`, `cat`, `grep`, etc. This guide assumes Git Bash throughout.

### Step 2: Activate Virtual Environment

```bash
# Activate the virtual environment
source .venv/Scripts/activate

# Verify activation (should show .venv path)
which python
python --version  # Should show Python 3.14.7 or similar
```

### Step 3: Generate Data (All Scenarios)

```bash
# Generate clean scenario
python scripts/generate_data.py --scenario clean

# Generate all scenarios (takes ~2 minutes)
python scripts/generate_data.py --scenario all

# Verify data generation
ls data/processed/
# Should see: clean_train.parquet, clean_val.parquet, clean_test.parquet, etc.
```

**Expected Output:**
- Training data: ~7,500 rows
- Validation data: ~1,250 rows  
- Test data: ~1,200 rows
- Total: 10,000 rows per scenario

### Step 4: Run Baseline Pipeline

```bash
# Run baseline ML pipeline (no quality gates)
python scripts/run_baseline.py --scenario clean

# Check the output
cat reports/baseline_run_summary.json
```

**Expected Results:**
- Execution time: 1-3 seconds
- RMSE: ~15-16
- R²: ~0.58-0.60
- Model saved to: `artifacts/baseline_model.pkl`

### Step 5: Run Proposed MLOps Pipeline

```bash
# Run proposed pipeline with quality gates and MLflow
python scripts/run_proposed_pipeline.py --scenario clean

# Check the output
cat reports/proposed_pipeline_summary.json
```

**Expected Results:**
- Execution time: 10-15 seconds (includes MLflow setup)
- Data validation: PASS
- Model quality gate: PASS (with adjusted threshold)
- Model registered: True
- MLflow run ID: (unique hash)
- Model version: 1
- Alias: champion

### Step 6: View MLflow UI (Optional but Recommended)

**Option A: Using Python**
```bash
# In a NEW terminal window
cd "/d/Personal Projects/reliableml"
source .venv/Scripts/activate
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# Keep this terminal open
# Open browser: http://localhost:5000
```

**Option B: Using Docker Compose**
```bash
# Start MLflow server in Docker
docker compose up -d mlflow

# Open browser: http://localhost:5000
```

**What to See in MLflow UI:**
- Experiments list
- Run details with metrics (RMSE, R², MAE, MAPE)
- Parameters logged
- Model artifacts
- Dataset fingerprint

### Step 7: Test Data Quality Failure Scenario

```bash
# Generate data with quality issues
python scripts/generate_data.py --scenario data_quality_failure

# Try to run proposed pipeline (should FAIL at data gate)
python scripts/run_proposed_pipeline.py --scenario data_quality_failure

# Expected: Pipeline ABORTED before training due to data quality gate failure
```

**Expected Output:**
```
Data quality gate FAILED and blocking is enabled. Pipeline aborted.
Status: ABORTED
Reason: Data quality gate failure (blocking)
```

### Step 8: Simulate Production Data

```bash
# Simulate clean production traffic
python scripts/simulate_production.py --scenario clean

# Simulate severe drift
python scripts/simulate_production.py --scenario severe_drift

# Check generated data
ls data/production/
# Should see: production_clean.parquet, production_severe_drift.parquet
```

### Step 9: Run Drift Monitoring

```bash
# Monitor clean production data (should show no drift)
python scripts/run_drift_monitoring.py --scenario clean

# Monitor severe drift scenario
python scripts/run_drift_monitoring.py --scenario severe_drift

# View HTML report
start reports/drift_report.html  # Opens in browser
```

**Expected Results:**

**Clean Scenario:**
- Decision: CONTINUE
- Drift detected: No
- Drift share: <20%

**Severe Drift Scenario:**
- Decision: RETRAIN_REQUIRED
- Drift detected: Yes
- Drift share: >50%
- Retraining request created

### Step 10: Test Reproducibility

```bash
# Run the same pipeline twice and compare
python scripts/reproduce_run.py --scenario clean

# Check reproducibility report
cat reports/reproducibility_report.json
```

**Expected Output:**
```
Overall Reproducible: ✓ PASSED
Dataset Fingerprint Match: True
Features Match: True
Metrics Match: True
Prediction Array Match: True
```

### Step 11: Export Thesis Comparison Metrics

```bash
# Generate comparison tables and summary
python scripts/export_thesis_metrics.py

# View outputs
cat reports/comparison_summary.md
cat reports/comparison_metrics.csv
```

### Step 12: Start Model Serving API

**Option A: Local Development Server**
```bash
# Start FastAPI server
uvicorn reliableml.service.app:app --host 0.0.0.0 --port 8000 --reload

# In browser, open: http://localhost:8000/docs
# You'll see interactive API documentation
```

**Option B: Docker Compose (Recommended)**
```bash
# Start both MLflow and API service
docker compose up --build

# Services available:
# - MLflow UI: http://localhost:5000
# - API docs: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
```

### Step 13: Test API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make a Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "store_001",
    "product_category": "Electronics",
    "promotion_flag": 1,
    "price": 149.99,
    "inventory_level": 250,
    "competitor_price": 159.99,
    "temperature": 22.5,
    "day_of_week": 5,
    "month": 7,
    "previous_day_sales": 85.0
  }'
```

**Expected Response:**
```json
{
  "prediction": 115.23,
  "model_name": "sales-demand-forecaster",
  "model_version": "1.0",
  "timestamp": "2026-09-05T23:15:00",
  "request_id": "uuid-here"
}
```

### Step 14: Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/reliableml --cov-report=term

# Expected: 33/33 tests passing ✅
```

### Step 15: Stop Services

```bash
# Stop Docker services
docker compose down

# Deactivate virtual environment
deactivate
```

---

## 🎯 Quick Commands Reference

### Data Generation
```bash
python scripts/generate_data.py --scenario clean
python scripts/generate_data.py --scenario all
```

### Pipeline Execution
```bash
python scripts/run_baseline.py --scenario clean
python scripts/run_proposed_pipeline.py --scenario clean
python scripts/run_proposed_pipeline.py --scenario data_quality_failure
```

### Production Simulation
```bash
python scripts/simulate_production.py --scenario clean
python scripts/simulate_production.py --scenario severe_drift
```

### Monitoring
```bash
python scripts/run_drift_monitoring.py --scenario clean
python scripts/run_drift_monitoring.py --scenario severe_drift
```

### Reproducibility & Metrics
```bash
python scripts/reproduce_run.py --scenario clean
python scripts/export_thesis_metrics.py
```

### Services
```bash
# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# FastAPI server
uvicorn reliableml.service.app:app --port 8000 --reload

# Docker Compose (both services)
docker compose up --build
docker compose down
```

### Testing
```bash
pytest tests/ -v
pytest tests/ --cov=src/reliableml
```

---

## 📊 Expected Outputs Location

| Output | Location |
|--------|----------|
| Generated data | `data/processed/*.parquet` |
| Reference data | `data/reference/reference_baseline.parquet` |
| Production data | `data/production/*.parquet` |
| Baseline summary | `reports/baseline_run_summary.json` |
| Proposed summary | `reports/proposed_pipeline_summary.json` |
| Data validation | `reports/data_validation_report.json` |
| Drift report (HTML) | `reports/drift_report.html` |
| Drift decision | `reports/drift_decision.json` |
| Reproducibility | `reports/reproducibility_report.json` |
| Comparison metrics | `reports/comparison_metrics.csv` |
| Comparison summary | `reports/comparison_summary.md` |
| MLflow database | `mlruns.db` |
| MLflow artifacts | `artifacts/mlruns/` |
| Baseline model | `artifacts/baseline_model.pkl` |

---

## 🔧 Troubleshooting

### Issue: Wrong Terminal (cmd.exe or PowerShell)
**Symptoms:** Commands like `ls`, `cat`, `source` don't work
```bash
# You see errors like:
# 'ls' is not recognized as an internal or external command
# 'cat' is not recognized as an internal or external command
```

**Solution:** Close the current terminal and open **Git Bash** instead
1. Close Command Prompt/PowerShell
2. Press Windows Key → Search "Git Bash" → Open it
3. Navigate: `cd "/d/Personal Projects/reliableml"`

### Issue: "Module not found"
```bash
# Reinstall in editable mode
pip install -e .
```

### Issue: "Permission denied" on ports
```bash
# Change ports in docker-compose.yml or use different ports:
mlflow ui --port 5001
uvicorn reliableml.service.app:app --port 8001
```

### Issue: MLflow database locked
```bash
# Stop all MLflow processes
pkill -f "mlflow"

# Remove lock files
rm mlruns.db-shm mlruns.db-wal

# Restart
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

### Issue: Docker containers won't start
```bash
# Check Docker is running
docker ps

# Rebuild containers
docker compose build --no-cache
docker compose up
```

### Issue: Tests failing
```bash
# Generate test data first
python scripts/generate_data.py --scenario clean --rows 500

# Run specific test
pytest tests/test_data_generator.py -v
```

---

## 🎓 Full Thesis Experiment Workflow

Run this complete sequence for thesis evaluation:

```bash
# 1. Setup
cd "/d/Personal Projects/reliableml"
source .venv/Scripts/activate

# 2. Generate all data
python scripts/generate_data.py --scenario all

# 3. Run baseline on all scenarios
for scenario in clean data_quality_failure mild_drift severe_drift performance_degradation; do
  echo "Running baseline: $scenario"
  python scripts/run_baseline.py --scenario $scenario
done

# 4. Run proposed pipeline on all scenarios
for scenario in clean data_quality_failure mild_drift severe_drift performance_degradation; do
  echo "Running proposed: $scenario"
  python scripts/run_proposed_pipeline.py --scenario $scenario
done

# 5. Simulate production for drift scenarios
for scenario in clean mild_drift severe_drift; do
  python scripts/simulate_production.py --scenario $scenario
  python scripts/run_drift_monitoring.py --scenario $scenario
done

# 6. Verify reproducibility
python scripts/reproduce_run.py --scenario clean

# 7. Export all metrics
python scripts/export_thesis_metrics.py

# 8. Done! Check reports/ directory for all outputs
ls -la reports/
```

**Total execution time:** ~15-20 minutes

---

## 📝 Using Makefile (Alternative)

If you prefer using Make commands:

```bash
make setup          # Setup environment
make data           # Generate all data
make baseline       # Run baseline pipeline
make proposed       # Run proposed pipeline
make simulate-severe-drift  # Simulate drift
make monitor-severe # Run drift monitoring
make reproduce      # Test reproducibility
make metrics        # Export metrics
make test           # Run test suite
make serve          # Start API server
make compose-up     # Start Docker services
make compose-down   # Stop Docker services
```

---

## ✅ Verification Checklist

After running the project, verify:

- [ ] Data generated in `data/processed/` (5 scenarios)
- [ ] Baseline model saved to `artifacts/baseline_model.pkl`
- [ ] Proposed pipeline logged to MLflow (check UI at localhost:5000)
- [ ] Model registered in MLflow Model Registry (version 1, alias: champion)
- [ ] Data validation report shows PASS for clean scenario
- [ ] Data validation report shows FAIL for data_quality_failure scenario
- [ ] Drift monitoring shows CONTINUE for clean scenario
- [ ] Drift monitoring shows RETRAIN_REQUIRED for severe_drift scenario
- [ ] Reproducibility report shows all checks PASSED
- [ ] Comparison metrics CSV generated
- [ ] API health endpoint returns 200 OK
- [ ] All 33 tests passing

---

## 🎉 Success Indicators

You've successfully run the project when you can:

1. ✅ Generate synthetic data with all 5 scenarios
2. ✅ Train models with both baseline and proposed pipelines
3. ✅ See the data quality gate block invalid data
4. ✅ See the model quality gate block poor models
5. ✅ View experiment runs in MLflow UI
6. ✅ Access the model via API endpoints
7. ✅ Detect drift in production data
8. ✅ Reproduce exact results across runs
9. ✅ Generate thesis comparison tables

---

**Need Help?** Check the full README.md or documentation in `docs/` folder.

**Questions?** 
- Make sure you're using **Git Bash** (not cmd.exe or PowerShell)
- All commands in this guide require Git Bash for Unix-style commands
- If you see `'ls' is not recognized`, you're in the wrong terminal - switch to Git Bash!
