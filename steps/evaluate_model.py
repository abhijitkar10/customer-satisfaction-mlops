import logging

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step

from src.config import REGISTERED_MODEL_NAME, configure_mlflow
from src.evaluation import evaluate_model as _evaluate

# Quality gate: only register the model if it explains at least this much
# variance on the held-out test set. Prevents shipping a bad model.
R2_THRESHOLD = 0.05


@step(enable_cache=False)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mlflow_run_id: str,
) -> Annotated[float, "r2"]:
    """Compute test-set metrics, log them to the training run,
    and register the model if it passes the quality gate."""
    configure_mlflow()
    metrics = _evaluate(model, X_test, y_test)
    logging.info("Test metrics: %s", metrics)

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(metrics)

    if metrics["r2"] >= R2_THRESHOLD:
        mlflow.register_model(
            model_uri=f"runs:/{mlflow_run_id}/model",
            name=REGISTERED_MODEL_NAME,
        )
        logging.info("Model registered as '%s'", REGISTERED_MODEL_NAME)
    else:
        logging.warning(
            "R2 %.4f below threshold %.2f - model NOT registered",
            metrics["r2"],
            R2_THRESHOLD,
        )
    return metrics["r2"]
