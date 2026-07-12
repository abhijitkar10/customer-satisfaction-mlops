"""Model definitions using the strategy pattern: every model exposes the same
train() interface, so the training step can swap models via config."""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """Common interface for all models."""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        ...


class LinearRegressionModel(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model


class RandomForestModel(Model):
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model


MODELS = {
    "linear_regression": LinearRegressionModel,
    "random_forest": RandomForestModel,
}
