"""Multi-seed experiment runner for baseline, proposed, and ablation pipelines."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from reliableml.config import deep_merge, load_config
from reliableml.pipelines.baseline import run_baseline_pipeline
from reliableml.pipelines.proposed import run_proposed_pipeline

Pipeline = Callable[[dict[str, Any]], dict[str, Any]]


def _json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str]:
    dependencies: dict[str, str] = {}
    for package in ("numpy", "pandas", "lightgbm", "mlflow", "pandera", "evidently"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            continue
    return dependencies


def _set_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.setdefault("project", {})["random_seed"] = seed
    result.setdefault("model", {}).setdefault("hyperparameters", {})["random_state"] = seed
    return result


def _metric_values(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("test_metrics") or summary.get("metrics") or {}
    return {name: float(value) for name, value in metrics.items() if isinstance(value, (int, float))}


def build_experiment_manifest(
    experiment_config: dict[str, Any],
    *,
    base_config_path: str | Path,
    scenario_config_path: str | Path,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest describing every requested run."""
    settings = experiment_config.get("experiment", experiment_config)
    seeds = [int(seed) for seed in settings.get("seeds", [42])]
    variants = settings.get("variants", [])
    if not variants:
        raise ValueError("Experiment configuration must define at least one variant")
    if not seeds:
        raise ValueError("Experiment configuration must define at least one seed")
    return {
        "manifest_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "dependencies": _dependency_versions(),
        "base_config": str(base_config_path),
        "base_config_sha256": _file_sha256(base_config_path),
        "scenario_config": str(scenario_config_path),
        "scenario_config_sha256": _file_sha256(scenario_config_path),
        "scenario": settings.get("scenario", "clean"),
        "seeds": seeds,
        "variants": [
            {
                "name": variant["name"],
                "pipeline": variant.get("pipeline", variant.get("type", "proposed")),
                "pipeline_config": variant.get("pipeline_config"),
                "overrides": variant.get("overrides", {}),
            }
            for variant in variants
        ],
    }


def aggregate_results(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize metric mean, standard deviation, and 95% confidence intervals."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in run_results:
        variant = result["variant"]
        grouped.setdefault(variant, {})
        for name, value in result.get("metrics", {}).items():
            grouped[variant].setdefault(name, []).append(float(value))

    summary: dict[str, Any] = {"variants": {}, "run_count": len(run_results)}
    for variant, metrics in grouped.items():
        summary["variants"][variant] = {}
        for name, values in metrics.items():
            n = len(values)
            mean = sum(values) / n
            variance = sum((value - mean) ** 2 for value in values) / (n - 1) if n > 1 else 0.0
            standard_error = math.sqrt(variance / n)
            summary["variants"][variant][name] = {
                "n": n,
                "mean": mean,
                "std": math.sqrt(variance),
                "min": min(values),
                "max": max(values),
                "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
                "values": values,
            }
    return summary


def run_experiment(
    experiment_config: dict[str, Any],
    *,
    base_config_path: str | Path = "configs/base.yaml",
    scenario_config_path: str | Path = "configs/scenarios.yaml",
    output_dir: str | Path | None = None,
    pipelines: dict[str, Pipeline] | None = None,
) -> dict[str, Any]:
    """Execute all variant/seed combinations and write manifest and result JSON files."""
    settings = experiment_config.get("experiment", experiment_config)
    manifest = build_experiment_manifest(
        experiment_config,
        base_config_path=base_config_path,
        scenario_config_path=scenario_config_path,
    )
    root = Path(output_dir or settings.get("output_dir", "./reports/experiments"))
    root.mkdir(parents=True, exist_ok=True)
    _json_dump(manifest, root / "manifest.json")
    pipelines = pipelines or {"baseline": run_baseline_pipeline, "proposed": run_proposed_pipeline}
    results: list[dict[str, Any]] = []

    for variant in manifest["variants"]:
        pipeline_name = variant["pipeline"]
        if pipeline_name not in pipelines:
            raise ValueError(f"Unknown pipeline '{pipeline_name}' for variant '{variant['name']}'")
        for seed in manifest["seeds"]:
            run_dir = root / "runs" / variant["name"] / f"seed_{seed}"
            pipeline_config = variant.get("pipeline_config") or f"configs/{pipeline_name}.yaml"
            config = load_config(
                base_config_path=base_config_path,
                pipeline_config_path=pipeline_config,
                scenario_config_path=scenario_config_path,
                scenario_name=manifest["scenario"],
            )
            config = _set_seed(deep_merge(config, variant.get("overrides", {})), seed)
            config.setdefault("tracking", {})["summary_file"] = str(run_dir / "pipeline_summary.json")
            config.setdefault("outputs", {})["pipeline_summary"] = str(run_dir / "pipeline_summary.json")
            config.setdefault("model_output", {})["save_path"] = str(run_dir / "model.pkl")
            try:
                pipeline_summary = pipelines[pipeline_name](config)
                run_result = {
                    "variant": variant["name"],
                    "pipeline": pipeline_name,
                    "seed": seed,
                    "status": pipeline_summary.get("status", "COMPLETED"),
                    "metrics": _metric_values(pipeline_summary),
                    "summary": pipeline_summary,
                }
            except Exception as error:
                run_result = {
                    "variant": variant["name"],
                    "pipeline": pipeline_name,
                    "seed": seed,
                    "status": "FAILED",
                    "metrics": {},
                    "error": f"{type(error).__name__}: {error}",
                }
            _json_dump(run_result, run_dir / "result.json")
            results.append(run_result)

    aggregate = aggregate_results(results)
    _json_dump(aggregate, root / "aggregate_summary.json")
    _write_result_tables(results, aggregate, root)
    return {"manifest": manifest, "runs": results, "aggregate": aggregate}


def _write_result_tables(
    results: list[dict[str, Any]], aggregate: dict[str, Any], output_dir: Path
) -> None:
    """Write tabular raw and aggregate results for thesis analysis."""
    run_rows = []
    for result in results:
        row = {
            "variant": result["variant"],
            "pipeline": result["pipeline"],
            "seed": result["seed"],
            "status": result["status"],
            "error": result.get("error", ""),
        }
        row.update(result.get("metrics", {}))
        run_rows.append(row)
    columns = sorted({key for row in run_rows for key in row})
    with (output_dir / "per_run_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(run_rows)

    aggregate_rows = []
    for variant, metrics in aggregate["variants"].items():
        for metric, values in metrics.items():
            aggregate_rows.append({"variant": variant, "metric": metric, **values})
    columns = sorted({key for row in aggregate_rows for key in row})
    with (output_dir / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregate_rows)
