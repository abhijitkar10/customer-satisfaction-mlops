"""Export the latest registered model to deployment/model.joblib.

The exported file is committed to the repo as a fallback, so the deployed
Streamlit app can always serve predictions even if the MLflow registry is
unreachable. Run this after a pipeline run that registered a new model.
"""

import sys
from pathlib import Path

import joblib
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BUNDLED_MODEL_PATH, REGISTERED_MODEL_NAME, configure_mlflow


def main() -> None:
    configure_mlflow()
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")
    BUNDLED_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, BUNDLED_MODEL_PATH, compress=3)
    size_mb = BUNDLED_MODEL_PATH.stat().st_size / 1e6
    print(f"Exported latest '{REGISTERED_MODEL_NAME}' to {BUNDLED_MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
