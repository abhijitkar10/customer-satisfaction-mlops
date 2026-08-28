"""Download the official Olist dataset from Kaggle and build the merged
training table.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
(9 relational CSVs: orders, payments, customers, items, products, reviews, ...)

The raw data is relational. Joining items and payments to orders produces a
FAN-OUT: an order with 3 items and 2 payments becomes 6 rows that all share
one review score. Training on that duplicates orders across the train/test
split and inflates the score, so this script aggregates back to ONE ROW PER
ORDER -- the grain that matches the label.
"""

from pathlib import Path

import kagglehub
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "olist_orders_merged.csv"

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# How each order-item / payment column collapses to order level.
# Money is summed (the order's total), product attributes are averaged
# (the "typical" item in the basket), payment terms take the max.
AGGREGATIONS = {
    "payment_sequential": "max",
    "payment_installments": "max",
    "payment_value": "sum",
    "price": "sum",
    "freight_value": "sum",
    "product_name_lenght": "mean",
    "product_description_lenght": "mean",
    "product_photos_qty": "mean",
    "product_weight_g": "mean",
    "product_length_cm": "mean",
    "product_height_cm": "mean",
    "product_width_cm": "mean",
}


def main() -> None:
    print("Downloading olistbr/brazilian-ecommerce from Kaggle...")
    raw_dir = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))

    orders = pd.read_csv(raw_dir / "olist_orders_dataset.csv")
    payments = pd.read_csv(raw_dir / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv")
    items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(raw_dir / "olist_products_dataset.csv")
    translation = pd.read_csv(raw_dir / "product_category_name_translation.csv")
    reviews = pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv")

    # Join the relational tables down to order-item level.
    df = (
        orders
        .merge(payments, on="order_id")
        .merge(customers, on="customer_id")
        .merge(items, on="order_id")
        .merge(products, on="product_id")
        .merge(translation, on="product_category_name", how="left")
        .merge(reviews[["order_id", "review_score"]], on="order_id")
    )
    print(f"Joined to {len(df):,} order-item rows ({df.order_id.nunique():,} orders)")

    # Collapse the fan-out: one row per order.
    per_order = df.groupby("order_id").agg(AGGREGATIONS)
    per_order["n_items"] = df.groupby("order_id")["order_item_id"].nunique()

    # Columns that are constant within an order: take the first value.
    constant_cols = DATE_COLUMNS + [
        "order_status",
        "customer_state",
        "product_category_name_english",
        "review_score",
    ]
    per_order = per_order.join(df.groupby("order_id")[constant_cols].first())

    merged = per_order.reset_index()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Aggregated to {len(merged):,} orders x {len(merged.columns)} cols -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
