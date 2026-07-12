"""Evaluation metrics for regression models."""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return MSE, RMSE and R2 for the model on the test set."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, predictions),
    }
