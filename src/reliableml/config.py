"""Configuration loading and management module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_project_root() -> Path:
    """Find the root directory of the project.

    Searches upward from the current file until pyproject.toml is found.

    Returns:
        Path to the project root directory.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "configs").exists():
            return current
        current = current.parent
    return Path.cwd()


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary with parsed configuration data.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML content cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_absolute():
        root = find_project_root()
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML from {path}: {e}") from e

    return data


def deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with dict2 taking precedence.

    Args:
        dict1: Base dictionary.
        dict2: Overriding dictionary.

    Returns:
        Merged dictionary.
    """
    result = dict(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    base_config_path: str | Path = "configs/base.yaml",
    pipeline_config_path: str | Path | None = None,
    scenario_config_path: str | Path = "configs/scenarios.yaml",
    scenario_name: str | None = None,
) -> dict[str, Any]:
    """Load and merge configuration files.

    Loads the base configuration, optionally merges a pipeline-specific configuration
    (baseline or proposed), and attaches the scenario configuration if specified.

    Args:
        base_config_path: Path to base.yaml.
        pipeline_config_path: Path to baseline.yaml or proposed.yaml.
        scenario_config_path: Path to scenarios.yaml.
        scenario_name: Optional scenario to select from scenarios.yaml.

    Returns:
        Merged configuration dictionary.
    """
    config = load_yaml(base_config_path)

    if pipeline_config_path:
        pipeline_config = load_yaml(pipeline_config_path)
        config = deep_merge(config, pipeline_config)

    if scenario_config_path:
        scenarios_data = load_yaml(scenario_config_path)
        config["all_scenarios"] = scenarios_data.get("scenarios", {})
        config["drift_monitoring_config"] = scenarios_data.get("drift_monitoring", {})

        if scenario_name:
            if scenario_name not in config["all_scenarios"]:
                raise ValueError(
                    f"Scenario '{scenario_name}' not found. "
                    f"Available scenarios: {list(config['all_scenarios'].keys())}"
                )
            config["selected_scenario"] = scenario_name
            config["scenario"] = config["all_scenarios"][scenario_name]
        else:
            config["selected_scenario"] = "clean"
            config["scenario"] = config["all_scenarios"].get("clean", {})

    return config
