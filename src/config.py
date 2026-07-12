"""Central MLflow configuration.

Tracking store resolution, in order:
1. MLFLOW_TRACKING_URI env var — set this to a remote server (e.g. DagsHub)
   and both the pipeline and the app use it. Auth comes from the standard
   MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD env vars.
2. Otherwise: local sqlite db at the project root (works offline).
"""

import os
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "customer_satisfaction"
REGISTERED_MODEL_NAME = "customer_satisfaction_model"

# Bundled fallback model for deployments with no reachable registry
# (exported by scripts/export_model.py, committed to the repo).
BUNDLED_MODEL_PATH = PROJECT_ROOT / "deployment" / "model.joblib"


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
