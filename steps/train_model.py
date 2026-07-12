from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step

from src.config import configure_mlflow
from src.model_dev import MODELS


@step(enable_cache=False)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "random_forest",
) -> Tuple[
    Annotated[RegressorMixin, "model"],
    Annotated[str, "mlflow_run_id"],
]:
    """Train the configured model and log it to MLflow.

    One MLflow run is opened per pipeline execution; the run id is passed
    downstream so the evaluation step logs metrics into the same run.
    """
    configure_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train_rows", len(X_train))

        model_strategy = MODELS[model_name]()
        for param, value in vars(model_strategy).items():
            mlflow.log_param(param, value)

        model = model_strategy.train(X_train, y_train)
        mlflow.sklearn.log_model(model, name="model")
        return model, run.info.run_id
