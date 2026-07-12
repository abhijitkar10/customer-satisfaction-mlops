"""Download the official Olist dataset from Kaggle and build the merged
training table.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
(9 relational CSVs: orders, payments, customers, items, products, reviews, ...)

The raw data is relational; this script joins it into one order-item-level
table with the review score attached, written to data/olist_orders_merged.csv.
"""

from pathlib import Path

import kagglehub
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "olist_orders_merged.csv"


def main() -> None:
    print("Downloading olistbr/brazilian-ecommerce from Kaggle...")
    raw_dir = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    print(f"Raw files in {raw_dir}:")
    for f in sorted(raw_dir.glob("*.csv")):
        print(f"  {f.name}")

    orders = pd.read_csv(raw_dir / "olist_orders_dataset.csv")
    payments = pd.read_csv(raw_dir / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv")
    items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(raw_dir / "olist_products_dataset.csv")
    translation = pd.read_csv(raw_dir / "product_category_name_translation.csv")
    reviews = pd.read_csv(raw_dir / "olist_order_reviews_dataset.csv")

    # Join the relational tables down to one row per order item.
    df = (
        orders
        .merge(payments, on="order_id")
        .merge(customers, on="customer_id")
        .merge(items, on="order_id")
        .merge(products, on="product_id")
        .merge(translation, on="product_category_name", how="left")
        .merge(
            reviews[["order_id", "review_score", "review_comment_message"]],
            on="order_id",
        )
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows x {len(df.columns)} cols to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
