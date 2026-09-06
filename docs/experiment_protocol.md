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
   - Simulate production data for each scenario
   - Run drift monitoring against reference
   - Record decisions and drift metrics

5. **Reproducibility Check**
   - Run proposed pipeline twice on clean scenario
   - Compare fingerprints, metrics, predictions

6. **Metric Export**
   - Generate comparison tables
   - Analyze results

### Controls

- **Random Seed**: Fixed at 42 for all data generation
- **Model Hyperparameters**: Identical across baseline and proposed
- **Hardware**: Same machine for all runs
- **Software Versions**: Pinned dependencies
- **Dataset Size**: Same row counts across scenarios

### Repetitions

- Each scenario run once per pipeline type
- Reproducibility check: 2 runs with identical config
- Total runs: 5 scenarios × 2 pipelines + 2 reproducibility = 12 runs

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
- Boolean checks: gate triggered correctly?
- Categorical: drift decision matches expected?

### Quantitative Comparison
- Model performance: mean difference across scenarios
- Overhead: percentage increase in pipeline duration
- Reproducibility: absolute difference in metrics

## Threats to Validity

### Internal Validity
- **Confound**: Different codepaths between baseline and proposed
- **Mitigation**: Share core training/evaluation functions

### External Validity
- **Synthetic Data**: May not reflect real-world complexity
- **Mitigation**: Use realistic feature relationships and scenarios
- **Generalization**: Results specific to tabular regression
- **Mitigation**: Document limitations clearly

### Construct Validity
- **Reliability Definition**: Blocking invalid models is one aspect
- **Mitigation**: Measure multiple reliability indicators
- **Reproducibility**: Numeric tolerance affects pass/fail
- **Mitigation**: Document tolerance values explicitly

## Ethical Considerations

- **Research only**: Not for production use
- **No real user data**: Synthetic dataset only
- **No proprietary data**: Public approach, open implementation

## Timeline

1. Implementation: Complete
2. Data generation: ~5 minutes
3. Baseline runs: ~2 minutes (5 scenarios)
4. Proposed runs: ~5 minutes (5 scenarios)
5. Drift monitoring: ~3 minutes (5 scenarios)
6. Reproducibility: ~2 minutes
7. Analysis and reporting: Manual

**Total experiment runtime: ~20 minutes**
