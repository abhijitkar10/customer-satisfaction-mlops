"""Data cleaning, feature engineering and splitting, kept separate from the
pipeline steps so it can be unit-tested and reused (e.g. by the Streamlit app)."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# Features describing the order itself: what the customer paid, what the
# product looks like, and how big the basket was.
ORDER_FEATURES = [
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
    "n_items",
]

# Delivery features engineered from the order timestamps. These are the
# strongest predictors by far: late orders average 2.27 stars vs 4.29 for
# on-time ones. They are only known AFTER delivery, which scopes the model
# to post-delivery triage rather than at-purchase prediction.
DELIVERY_FEATURES = [
    "delivery_days",      # purchase -> delivered
    "delay_vs_promise",   # delivered - estimated (positive = LATE)
    "is_late",            # delivered after the promised date
    "approval_hours",     # purchase -> payment approved
]

FEATURE_COLUMNS = ORDER_FEATURES + DELIVERY_FEATURES

TARGET_COLUMN = "review_score"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive delivery-timing features from the raw order timestamps."""
    df = df.copy()
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    purchase = df["order_purchase_timestamp"]
    delivered = df["order_delivered_customer_date"]
    estimated = df["order_estimated_delivery_date"]

    df["delivery_days"] = (delivered - purchase).dt.days
    df["delay_vs_promise"] = (delivered - estimated).dt.days
    df["is_late"] = (df["delay_vs_promise"] > 0).astype(float)
    df["approval_hours"] = (
        df["order_approved_at"] - purchase
    ).dt.total_seconds() / 3600
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features, restrict to delivered orders, and impute the rest.

    Orders that were never delivered (~2% -- cancelled, still shipping) are
    dropped rather than imputed: they are a different failure mode with an
    average score of 1.76 and are better handled by a business rule than by
    this model, which predicts satisfaction *given* delivery.
    """
    df = engineer_features(df)

    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN, "order_delivered_customer_date"])
    logging.info("Dropped %d undelivered/unlabelled orders", before - len(df))

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for col in FEATURE_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    logging.info("Cleaned data: %d rows, %d features", len(df), len(FEATURE_COLUMNS))
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets. Returns X_train, X_test, y_train, y_test."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
