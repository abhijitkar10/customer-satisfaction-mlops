from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import clean_data as _clean, split_data


@step
def clean_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Clean the raw data and split it into train/test sets."""
    cleaned = _clean(df)
    X_train, X_test, y_train, y_test = split_data(cleaned)
    return X_train, X_test, y_train, y_test
