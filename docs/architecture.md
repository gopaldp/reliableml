# System Architecture

## Overview

ReliableML implements a production-grade MLOps architecture comparing a traditional baseline approach against a proposed system with automated quality gates and drift monitoring.

## Component Architecture

### 1. Data Layer

**Data Generator**
- Deterministic synthetic data generation using numpy random generators with fixed seeds
- Configurable scenarios injecting quality issues and distribution shifts
- SHA-256 fingerprinting for lineage tracking

**Data Validation (Proposed Only)**
- Pandera schema enforcement (types, ranges, categorical constraints)
- Custom quality gates (missing ratio, duplicates, domain rules)
- Blocking pipeline execution on validation failure

### 2. Training Layer

**Preprocessor**
- Categorical encoding (label encoding with unknown category handling)
- Missing value imputation (median for numerics)
- Feature engineering (price differences, ratios)
- Fitted on training data, transforms validation/test consistently

**Model Training**
- LightGBM gradient boosting regressor
- Hyperparameters configurable via YAML
- Training metadata tracked (duration, trees, features)

**Evaluation**
- Regression metrics: MAE, RMSE, R², MAPE
- Inference latency measurement (per-record and batch)
- Model quality gates with configurable thresholds

### 3. Tracking & Registry Layer (Proposed Only)

**MLflow Tracking**
- SQLite backend for run metadata
- Local file system for artifacts
- Logs: parameters, metrics, model, artifacts, code version, environment

**MLflow Model Registry**
- Registered model with versions
- Alias-based promotion (`champion` for production)
- Model metadata and lineage

### 4. Serving Layer

**FastAPI Application**
- `/health` - Service and model status
- `/model-info` - Detailed model metadata
- `/predict` - Single prediction with validation
- `/batch-predict` - Batch predictions

**Prediction Logging**
- JSONL append-only log
- Captures features, prediction, timestamp, model version
- Enables offline drift detection

### 5. Monitoring Layer (Proposed Only)

**Drift Detection (Evidently)**
- Kolmogorov-Smirnov test for numerical features
- Chi-square test for categorical features
- HTML visualization reports
- JSON structured results

**Decision Engine**
- CONTINUE: No significant drift
- INVESTIGATE: Moderate drift, manual review recommended
- RETRAIN_REQUIRED: Severe drift, automatic retraining trigger

### 6. Containerization

**Docker Services**
- `mlflow`: Tracking server (port 5000)
- `prediction-service`: FastAPI model serving (port 8000)
- Shared volumes for database, artifacts, data, logs

## Data Flow

### Baseline Pipeline
```
Raw Data → Train Model → Evaluate → Save Pickle → Done
```

### Proposed Pipeline
```
Raw Data → Quality Gate → [PASS] → Train Model → 
Evaluate → Quality Gate → [PASS] → MLflow Registry → 
FastAPI Service → Predictions → Drift Monitor → Decision
```

## Quality Gates

### Data Quality Gate
- **Trigger**: Before model training
- **Checks**: Schema, missing values, duplicates, domain constraints
- **Action**: FAIL = halt pipeline, PASS = continue

### Model Quality Gate
- **Trigger**: After evaluation, before registry
- **Checks**: RMSE, MAPE, R², inference latency
- **Action**: FAIL = block registration, PASS = register model

### Drift Monitoring Gate
- **Trigger**: Offline, on production data
- **Checks**: Feature distribution shifts
- **Action**: RETRAIN_REQUIRED = trigger retraining pipeline

## Technology Stack Rationale

- **LightGBM**: Fast, handles categorical features natively, good for tabular data
- **Pandera**: DataFrame validation with clear error messages
- **MLflow**: Industry-standard experiment tracking and model registry
- **Evidently**: Statistical drift detection with visualization
- **FastAPI**: Modern Python API framework with automatic OpenAPI docs
- **Docker Compose**: Simple local multi-service orchestration

## Scalability Considerations

This is a **local prototype** optimized for:
- Single-machine execution
- Small to medium datasets (up to 100k rows)
- Development and thesis demonstration

**Production considerations not implemented:**
- Distributed training (Spark, Ray)
- Cloud object storage (S3, GCS)
- Workflow orchestration (Airflow, Prefect)
- Real-time streaming (Kafka)
- Horizontal scaling (Kubernetes)
- Monitoring dashboards (Grafana)

## Security

- Non-root container users
- Input validation with Pydantic
- No secrets in code (environment variables)
- Health checks for service monitoring
- Secure defaults (HTTPS in production recommended)
