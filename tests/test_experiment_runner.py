"""Focused tests for the multi-seed experiment runner."""

from __future__ import annotations

import json

from reliableml.experiments.runner import aggregate_results, run_experiment


def test_aggregate_results_reports_sample_statistics() -> None:
    result = aggregate_results(
        [
            {"variant": "baseline", "metrics": {"rmse": 2.0}},
            {"variant": "baseline", "metrics": {"rmse": 4.0}},
        ]
    )

    metric = result["variants"]["baseline"]["rmse"]
    assert metric["n"] == 2
    assert metric["mean"] == 3.0
    assert metric["std"] == 2**0.5
    assert metric["ci95"][0] < 3 < metric["ci95"][1]


def test_run_experiment_writes_manifest_and_per_run_results(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    pipeline = tmp_path / "pipeline.yaml"
    scenarios = tmp_path / "scenarios.yaml"
    base.write_text("project:\n  random_seed: 1\nmodel:\n  hyperparameters: {}\n", encoding="utf-8")
    pipeline.write_text("pipeline:\n  type: baseline\n", encoding="utf-8")
    scenarios.write_text("scenarios:\n  clean: {}\n", encoding="utf-8")

    observed_seeds: list[int] = []

    def fake_pipeline(config):
        observed_seeds.append(config["project"]["random_seed"])
        return {"metrics": {"rmse": float(config["project"]["random_seed"])}}

    config = {
        "experiment": {
            "scenario": "clean",
            "seeds": [7, 11],
            "variants": [
                {
                    "name": "smoke",
                    "pipeline": "baseline",
                    "pipeline_config": str(pipeline),
                }
            ],
        }
    }
    result = run_experiment(
        config,
        base_config_path=base,
        scenario_config_path=scenarios,
        output_dir=tmp_path / "results",
        pipelines={"baseline": fake_pipeline},
    )

    assert observed_seeds == [7, 11]
    assert (tmp_path / "results" / "manifest.json").exists()
    assert (tmp_path / "results" / "per_run_results.csv").exists()
    assert (tmp_path / "results" / "aggregate_results.csv").exists()
    run_file = tmp_path / "results" / "runs" / "smoke" / "seed_7" / "result.json"
    assert json.loads(run_file.read_text(encoding="utf-8"))["metrics"] == {"rmse": 7.0}
    assert result["aggregate"]["variants"]["smoke"]["rmse"]["n"] == 2
