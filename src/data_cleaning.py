"""Data cleaning and preprocessing logic, kept separate from pipeline steps
so it can be unit-tested and reused (e.g. by the Streamlit app)."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# Numeric features describing the order: what the customer paid, what the
# product looks like, and how it was shipped.
FEATURE_COLUMNS = [
    "payment_sequential",
    "payment_installments",
    "payment_value",
    "price",
    "freight_value",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

TARGET_COLUMN = "review_score"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features, drop rows without a review score,
    and impute missing numeric values with the median."""
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df = df.dropna(subset=[TARGET_COLUMN])
    for col in FEATURE_COLUMNS:
        df[col] = df[col].fillna(df[col].median())
    logging.info("Cleaned data: %d rows, %d features", len(df), len(FEATURE_COLUMNS))
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets. Returns X_train, X_test, y_train, y_test."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
