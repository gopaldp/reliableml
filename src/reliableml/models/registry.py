"""Model registry utilities for MLflow."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    tags: dict[str, Any] | None = None,
) -> str:
    """Register a model in MLflow Model Registry.

    Args:
        run_id: MLflow run ID containing the model.
        model_name: Name for the registered model.
        artifact_path: Path to the model artifact within the run.
        tags: Optional tags to apply to the model version.

    Returns:
        Model version string.
    """
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Registering model from {model_uri}")

    try:
        model_version = mlflow.register_model(model_uri, model_name)
        version_number = model_version.version

        # Apply tags if provided
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(model_name, version_number, key, str(value))

        logger.info(f"Model registered: {model_name} version {version_number}")
        return str(version_number)

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise


def set_model_alias(
    model_name: str,
    version: str,
    alias: str,
) -> None:
    """Set an alias for a model version (e.g., 'champion', 'challenger').

    Args:
        model_name: Registered model name.
        version: Model version number.
        alias: Alias name to set.
    """
    client = MlflowClient()

    try:
        client.set_registered_model_alias(model_name, alias, version)
        logger.info(f"Set alias '{alias}' for {model_name} version {version}")
    except Exception as e:
        logger.error(f"Failed to set model alias: {e}")
        raise


def get_model_by_alias(
    model_name: str,
    alias: str,
) -> Any:
    """Load a model by its alias.

    Args:
        model_name: Registered model name.
        alias: Alias name (e.g., 'champion').

    Returns:
        Loaded model object.
    """
    model_uri = f"models:/{model_name}@{alias}"
    logger.info(f"Loading model from {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Failed to load model by alias: {e}")
        raise


def get_model_metadata(
    model_name: str,
    alias: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Retrieve metadata for a registered model.

    Args:
        model_name: Registered model name.
        alias: Optional alias to resolve to a version.
        version: Optional explicit version number.

    Returns:
        Dictionary containing model metadata.
    """
    client = MlflowClient()

    if alias and not version:
        # Resolve alias to version
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
            version = model_version.version
        except Exception as e:
            logger.error(f"Failed to resolve alias '{alias}': {e}")
            raise

    if not version:
        raise ValueError("Either alias or version must be provided")

    try:
        model_version = client.get_model_version(model_name, version)
        run = client.get_run(model_version.run_id)

        metadata = {
            "model_name": model_name,
            "version": version,
            "run_id": model_version.run_id,
            "status": model_version.status,
            "tags": model_version.tags,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "description": model_version.description,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }

        return metadata

    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
        raise


def list_registered_models(name_prefix: str | None = None) -> list[str]:
    """List registered model names.

    Args:
        name_prefix: Optional prefix filter.

    Returns:
        List of registered model names.
    """
    client = MlflowClient()
    models = client.search_registered_models()

    model_names = [model.name for model in models]

    if name_prefix:
        model_names = [name for name in model_names if name.startswith(name_prefix)]

    return model_names
