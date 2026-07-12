import argparse

from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["linear_regression", "random_forest"],
        help="Which model to train",
    )
    parser.add_argument(
        "--data",
        default="data/olist_orders_merged.csv",
        help="Path to the raw dataset",
    )
    args = parser.parse_args()

    training_pipeline(data_path=args.data, model_name=args.model)
