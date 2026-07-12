import logging

import pandas as pd
from zenml import step


# Cache disabled: the CSV can be re-downloaded/changed without the path
# changing, so the path alone is not a safe cache key.
@step(enable_cache=False)
def ingest_data(data_path: str) -> pd.DataFrame:
    """Read the raw Olist orders dataset from CSV."""
    df = pd.read_csv(data_path)
    logging.info("Ingested %d rows from %s", len(df), data_path)
    return df
