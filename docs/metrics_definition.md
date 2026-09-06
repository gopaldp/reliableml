# Metrics Definition

## Model Performance Metrics

### MAE (Mean Absolute Error)
**Definition**: Average of absolute differences between predictions and actual values.

```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Interpretation**:
- Lower is better
- Same units as target variable (sales demand)
- Robust to outliers
- **Typical Range**: 10-50 for this dataset

### RMSE (Root Mean Squared Error)
**Definition**: Square root of average squared differences.

```
RMSE = √[(1/n) * Σ(y_true - y_pred)²]
```

**Interpretation**:
- Lower is better
- Penalizes large errors more than MAE
- Same units as target
- **Quality Gate Threshold**: < 150

### R² (Coefficient of Determination)
**Definition**: Proportion of variance explained by the model.

```
R² = 1 - (SS_res / SS_tot)
```

**Interpretation**:
- Range: (-∞, 1], typically [0, 1]
- Higher is better
- 1.0 = perfect fit, 0.0 = no better than mean
- **Quality Gate Threshold**: > 0.6

### MAPE (Mean Absolute Percentage Error)
**Definition**: Average of absolute percentage errors.

```
MAPE = (1/n) * Σ|((y_true - y_pred) / y_true)| * 100%
```

**Interpretation**:
- Lower is better
- Scale-independent (percentage)
- Undefined when y_true = 0
- **Quality Gate Threshold**: < 25%

## Pipeline Performance Metrics

### Training Duration (seconds)
**Definition**: Wall-clock time for model training only.

**Measured**: Start of `train_lightgbm_model()` to model return.

**Typical Range**: 1-10 seconds for small datasets.

### Total Pipeline Duration (seconds)
**Definition**: End-to-end execution time.

**Includes**:
- Data loading
- Validation (proposed only)
- Preprocessing
- Training
- Evaluation
- Registration (proposed only)
- Artifact saving

**Expected Overhead**: Proposed pipeline ~2-3x longer due to gates and logging.

### Inference Latency (milliseconds)
**Per-Record Latency**: Average time to predict one sample.

**Measured**: 100 iterations, warmup excluded, per-record calculation.

**Quality Gate Threshold**: < 100 ms per record.

## Data Quality Metrics

### Missing Ratio
**Definition**: Proportion of missing cells in the dataset.

```
Missing Ratio = (Total Missing Cells) / (Total Cells)
```

**Quality Gate Threshold**: < 5%

### Duplicate Ratio
**Definition**: Proportion of duplicate rows.

```
Duplicate Ratio = (Duplicate Rows) / (Total Rows)
```

**Quality Gate Threshold**: < 1%

### Dataset Fingerprint
**Definition**: SHA-256 hash of sorted CSV representation.

**Purpose**: Deterministic identifier for exact dataset reproduction.

## Drift Metrics

### Drift Share
**Definition**: Proportion of features exhibiting significant drift.

```
Drift Share = (Number of Drifted Features) / (Total Features)
```

**Decision Thresholds**:
- < 20%: CONTINUE
- 20-50%: INVESTIGATE
- > 50%: RETRAIN_REQUIRED

### P-Value (Statistical Tests)
**Kolmogorov-Smirnov**: Numerical feature distributions.

**Chi-Square**: Categorical feature distributions.

**Significance Level**: 0.05 (95% confidence).

## Reliability Metrics

### Invalid Release Prevented (boolean)
**Definition**: Did quality gates correctly block a bad model?

**Measurement**:
- `data_quality_failure` scenario: Should block at data gate
- Low-quality model: Should block at model gate

**Expected**:
- Baseline: False (no gates)
- Proposed: True (gates active)

### Model Registered (boolean)
**Definition**: Was model successfully registered in MLflow?

**Depends On**:
- Data quality gate: PASS
- Model quality gate: PASS

### Deployable (string)
**Values**:
- "Yes (Verified)": Passed all gates
- "No (Blocked)": Failed gates
- "Yes (Unverified)": No gates (baseline)

## Reproducibility Metrics

### Fingerprint Match (boolean)
**Definition**: Identical dataset hashes across runs.

**Expected**: True for identical seed and config.

### Metrics Match (boolean)
**Definition**: Model metrics within tolerance.

**Tolerance**:
- MAE: ±0.01
- RMSE: ±0.01
- R²: ±0.001
- MAPE: ±0.001

### Predictions Match (boolean)
**Definition**: Prediction arrays within tolerance.

**Tolerance**: ±1e-5 (floating-point precision).

## Comparison Metrics

### Reliability Score
**Formula**: (Invalid Releases Prevented) / (Total Failure Scenarios)

**Example**: 1/1 = 100% for proposed, 0/1 = 0% for baseline.

### Reproducibility Score
**Formula**: (Checks Passed) / (Total Checks)

**Checks**: Fingerprint, Features, Metrics, Predictions.

### Operational Overhead
**Formula**: (Proposed Duration - Baseline Duration) / (Baseline Duration)

**Acceptable Range**: < 3x (less than 200% overhead).

## Thesis-Specific Metrics

### Experiment Tracking Present (boolean)
- Baseline: False
- Proposed: True (MLflow)

### Dataset Fingerprint Tracked (boolean)
- Baseline: False
- Proposed: True

### Drift Detected (boolean)
Per scenario, whether drift was detected.

### Decision Correctness (boolean)
Did drift decision match expected outcome?

## Aggregated Reporting

### Comparison Table Columns
1. pipeline_type
2. scenario
3. data_quality_gate_status
4. model_quality_gate_status
5. model_registered
6. deployable
7. drift_detected
8. decision
9. mae, rmse, r2, mape
10. train_duration_s
11. total_duration_s
12. inference_latency_ms
13. dataset_fingerprint_tracked
14. experiment_tracking_present
15. reproducibility_verified
16. invalid_release_prevented

### Summary Statistics
- Mean, median, std per metric
- Pass/fail rates per gate
- Drift detection accuracy matrix
