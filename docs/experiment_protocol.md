# Experimental Protocol

## Research Question

**How can automated data-quality gates and data-drift monitoring improve the reliability and reproducibility of continuous machine-learning deployment compared with a conventional CI/CD pipeline?**

## Hypotheses

### H1: Reliability
Automated data-quality gates prevent invalid models from reaching production, reducing deployment failures compared to pipelines without validation.

### H2: Reproducibility
Dataset fingerprinting and experiment tracking enable exact reproduction of model training results, which is not possible with ad-hoc workflows.

### H3: Operational Safety
Automated drift detection triggers timely retraining, maintaining model performance under distribution shifts better than manual monitoring.

## Independent Variables

1. **Pipeline Type** (categorical)
   - Baseline: Traditional ML workflow
   - Proposed: MLOps workflow with gates and monitoring

2. **Data Scenario** (categorical)
   - `clean`: Normal operation
   - `data_quality_failure`: Schema violations, missing values
   - `mild_drift`: Minor distribution shifts
   - `severe_drift`: Major distribution changes
   - `performance_degradation`: Concept drift

## Dependent Variables

### Primary Metrics
1. **Invalid Release Prevention** (binary)
   - Did quality gates block a bad model from deployment?
   
2. **Reproducibility Success** (binary)
   - Can identical results be reproduced with same seed/config?
   
3. **Drift Detection Accuracy** (categorical)
   - Correct decision: CONTINUE / INVESTIGATE / RETRAIN_REQUIRED

### Secondary Metrics
4. **Model Performance** (continuous)
   - MAE, RMSE, R², MAPE
   
5. **Pipeline Overhead** (continuous)
   - Training duration, total pipeline duration
   
6. **Lineage Completeness** (ordinal)
   - Dataset fingerprint tracked? (yes/no)
   - Experiment metadata logged? (yes/no)

## Experimental Design

The evaluation combines a controlled synthetic benchmark with two public
demand datasets. The synthetic benchmark provides known failure and drift
ground truth; the public datasets test whether conclusions transfer beyond the
data generator.

### Dataset Tracks

1. **Synthetic sales demand**: controlled clean, quality-failure, mild-drift,
   severe-drift, and performance-degradation scenarios.
2. **Public dataset A - UCI Online Retail**: transaction-level retail data
   aggregated into a documented demand-prediction task.
3. **Public dataset B - UCI Bike Sharing**: hourly demand forecasting with
   documented temporal and weather covariates.

Each public dataset must have a version or download checksum recorded in the
experiment manifest. Dataset-specific adapters map source columns into the
common feature/target contract without changing pipeline logic.

### Procedure

1. **Setup**
   - Generate all scenario datasets with fixed seed (42)
   - Create reference baseline from clean training data

2. **Baseline Runs**
   - Execute baseline pipeline on all 5 scenarios
   - Record metrics and artifacts

3. **Proposed Runs**
   - Execute proposed pipeline on all 5 scenarios
   - Record metrics, MLflow runs, quality gate results

4. **Drift Monitoring**
   - Evaluate clean, mild, and severe shifts with known labels
   - Run drift monitoring against the training reference
   - Record decisions, drift metrics, false alarms, missed detections, and
     detection latency

5. **Ablation Runs**
   - Baseline only
   - Baseline plus data-quality gates
   - Baseline plus drift monitoring
   - Baseline plus tracking and fingerprinting
   - Complete proposed pipeline
   - Keep model, data split, and hyperparameters fixed within each comparison

6. **Reproducibility Check**
   - Repeat the complete proposed pipeline with identical inputs
   - Repeat selected experiments across independent random seeds and a clean
     environment rebuild
   - Compare fingerprints, metrics, predictions, and serialized manifests

7. **Metric Export**
   - Generate comparison tables
   - Analyze results

### Controls

- **Random Seed**: Fixed at 42 for all data generation
- **Model Hyperparameters**: Identical across baseline and proposed
- **Hardware**: Same machine for all runs
- **Software Versions**: Lock-file or generated environment manifest, including
  operating system, Python version, and library versions
- **Dataset Size**: Same row counts across scenarios
- **Data Splits**: Chronological train/validation/test splits for demand data;
  no random future leakage

### Repetitions

- Minimum 10 independent seeds for each primary synthetic comparison
- Minimum 5 repetitions per public-dataset pipeline/ablation condition
- At least 2 identical-input runs for deterministic reproducibility checks
- Report every run, not only aggregate values
- Record the final run count in the generated experiment manifest

## Expected Outcomes

### H1 Validation
- **Baseline**: `data_quality_failure` scenario produces model (no gate)
- **Proposed**: Pipeline halts before training (gate blocks)
- **Evidence**: Pipeline status, model registration boolean

### H2 Validation
- **Baseline**: Fingerprint missing, no run tracking
- **Proposed**: Identical fingerprints, metrics within tolerance
- **Evidence**: Reproducibility report JSON

### H3 Validation
- **Baseline**: No drift detection, manual intervention required
- **Proposed**: Correct decisions (CONTINUE for clean, RETRAIN for severe)
- **Evidence**: Drift decision JSON, accuracy per scenario

## Data Collection

All outputs saved to `reports/`:
- `baseline_run_summary.json`
- `proposed_pipeline_summary.json`
- `data_validation_report.json`
- `drift_decision.json`
- `reproducibility_report.json`
- `comparison_metrics.csv`
- `comparison_summary.md`

## Statistical Analysis

### Qualitative Comparison
- Confusion matrix for drift decisions against scenario ground truth
- Gate decision matrix: valid release accepted, invalid release blocked,
  invalid release accepted, and valid release blocked
- Failure taxonomy for schema, missingness, drift, and model-quality failures

### Quantitative Comparison
- Model performance: mean, median, standard deviation, and 95% confidence
  interval for MAE, RMSE, R2, and MAPE
- Operational overhead: paired proposed-vs-baseline runtime ratio and
  bootstrap confidence interval
- Reliability: blocked-invalid-release rate, false-block rate, and
  release-decision accuracy
- Drift monitoring: precision, recall, F1, false-alarm rate, missed-drift rate,
  and detection delay
- Reproducibility: maximum and distribution of metric/prediction differences

For paired repeated measurements, use a paired permutation test or Wilcoxon
signed-rank test when normality is not justified. Report effect sizes and
confidence intervals, not only p-values. Correct for multiple primary
comparisons or label secondary analyses as exploratory.

### Pre-registered Acceptance Criteria

The hypotheses are supported only if the results meet criteria defined before
the final runs:

- **H1**: the proposed pipeline blocks at least 95% of injected invalid
  releases while keeping the valid-release false-block rate below 5%
- **H2**: identical-input runs reproduce all fingerprints and remain within
  documented metric and prediction tolerances
- **H3**: drift detection achieves at least 90% recall on severe drift and
  reports a measured operational false-alarm rate on clean data
- **Overhead constraint**: median proposed pipeline overhead remains below 3x
  the matched baseline unless a higher cost is explicitly justified

These thresholds are evaluation criteria, not guaranteed outcomes.

## Threats to Validity

### Internal Validity
- **Confound**: Different codepaths between baseline and proposed
- **Mitigation**: Share core training/evaluation functions

### External Validity
- **Synthetic Data**: May not reflect real-world complexity
- **Mitigation**: Use realistic feature relationships and scenarios
- **Generalization**: Results specific to tabular regression
- **Mitigation**: Evaluate two public demand datasets with different feature
  distributions and document remaining limitations clearly
- **Dataset selection bias**: Public datasets may be unusually clean or
  convenient
- **Mitigation**: Publish inclusion criteria, preprocessing decisions, and
  failed or unsupported dataset attempts
- **Concept-drift ground truth**: Real-world drift labels are rarely known
- **Mitigation**: Use injected shifts with known labels for controlled
  evaluation and report real-data drift as exploratory
- **Threshold tuning bias**: Gate and drift thresholds can be selected to fit
  observed scenarios
- **Mitigation**: Freeze thresholds using a calibration split before final
  test runs

### Construct Validity
- **Reliability Definition**: Blocking invalid models is one aspect
- **Mitigation**: Measure multiple reliability indicators
- **Reproducibility**: Numeric tolerance affects pass/fail
- **Mitigation**: Document tolerance values explicitly

## Ethical Considerations

- **Research only**: Not for production use
- **No real user data**: Public datasets only; no private user data is used
- **No proprietary data**: Public approach, open implementation, and dataset
  licenses documented in the replication package

## Timeline

1. Implementation and unit tests: Complete
2. Dataset adapters and checksums
3. Calibration and pilot runs
4. Locked repeated experiment matrix
5. Ablation and transfer evaluation
6. Statistical analysis and plots
7. Thesis writing, review, and replication package

The runtime is reported from the generated manifest after the experiment matrix
is finalized; it is not assumed to be 20 minutes once repetitions and public
datasets are included.
