from zenml import pipeline

from steps.clean_data import clean_data
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.train_model import train_model


@pipeline
def training_pipeline(data_path: str, model_name: str = "random_forest"):
    """Ingest -> clean -> train -> evaluate."""
    df = ingest_data(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_data(df=df)
    model, run_id = train_model(
        X_train=X_train, y_train=y_train, model_name=model_name
    )
    evaluate_model(
        model=model, X_test=X_test, y_test=y_test, mlflow_run_id=run_id
    )
