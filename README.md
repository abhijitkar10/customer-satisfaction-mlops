# Customer Satisfaction Prediction Pipeline

An end-to-end MLOps pipeline that predicts the review score (1–5) a customer
will leave for an e-commerce order, built on the real
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from Kaggle (~100k orders, 9 relational tables).

## Data

The Kaggle dataset ships as 9 normalized CSVs (orders, payments, customers,
order items, products, reviews, ...). [scripts/download_data.py](scripts/download_data.py)
downloads them via `kagglehub` (no API key needed) and joins them into a
single order-item-level table (117k rows × 33 cols) with the review score
attached — the input to the training pipeline.

**Stack:** Python · ZenML (orchestration) · MLflow (experiment tracking + model
registry) · Scikit-learn (models) · Streamlit (serving UI)

## Architecture

```
                        ZenML training pipeline
 ┌─────────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────────┐
 │ ingest_data │──▶│ clean_data │──▶│ train_model │──▶│ evaluate_model │
 └─────────────┘   └────────────┘   └─────────────┘   └────────────────┘
   read merged       select 12          fit model,        MSE/RMSE/R²,
   CSV (117k rows)   features,          log params +      register model
                     impute NAs,        model to MLflow   only if R² gate
                     80/20 split                          passes
                                            │                   │
                                            ▼                   ▼
                                    ┌──────────────────────────────┐
                                    │   MLflow (sqlite backend)    │
                                    │  runs · metrics · registry   │
                                    └──────────────┬───────────────┘
                                                   │ models:/…/latest
                                                   ▼
                                        ┌─────────────────────┐
                                        │    Streamlit app    │
                                        │  interactive scorer │
                                        └─────────────────────┘
```

Key design points:

- **One MLflow run per pipeline execution.** The training step opens the run
  and passes its `run_id` downstream, so evaluation metrics land in the same
  run as the model and its parameters.
- **Quality gate before registration.** A model is only pushed to the MLflow
  registry if test-set R² clears a threshold — the Streamlit app can never
  silently pick up a bad model.
- **Strategy pattern for models** ([src/model_dev.py](src/model_dev.py)):
  every model exposes the same `train()` interface, so swapping models is a
  CLI flag, not a code change.
- **Cleaning logic lives in [src/](src/), not in pipeline steps**, so it is
  unit-testable and reusable by the app.

## Results

| Model             | RMSE  | R²    | Registered? |
|-------------------|-------|-------|-------------|
| Linear regression | 1.385 | 0.017 | ❌ (below gate) |
| Random forest     | 1.333 | 0.089 | ✅ |

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt   # full training deps
zenml init

# Download the Olist dataset from Kaggle and build the merged table
python scripts/download_data.py

# Train (each run is tracked in MLflow)
python run_pipeline.py --model random_forest
python run_pipeline.py --model linear_regression

# Compare runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Serve predictions from the latest registered model
streamlit run app.py
```

## Deployment (all free tiers)

```
GitHub (code) ──▶ Streamlit Community Cloud (live demo app)
      │
      └─────────▶ DagsHub (hosted MLflow: public runs, metrics, registry)
```

- **MLflow store is env-driven** ([src/config.py](src/config.py)): set
  `MLFLOW_TRACKING_URI` (+ `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`)
  to log to a remote server such as DagsHub; unset, everything falls back to
  local sqlite and works offline.
- **The app prefers the registry, falls back to a bundled model**
  ([deployment/model.joblib](deployment/model.joblib), exported via
  `python scripts/export_model.py`) — a live demo can never break because the
  tracking server is down.
- **Serving deps are split from training deps**: Streamlit Cloud installs the
  slim root [requirements.txt](requirements.txt) (no ZenML — inference doesn't
  need an orchestrator); training uses
  [requirements-pipeline.txt](requirements-pipeline.txt).

To log training runs to DagsHub instead of local sqlite:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
python run_pipeline.py
```

## Project layout

```
├── data/                    # merged Olist table (built from Kaggle raw CSVs)
├── deployment/
│   └── model.joblib         # bundled fallback model for the deployed app
├── scripts/
│   ├── download_data.py     # Kaggle download + relational join
│   └── export_model.py      # registry → deployment/model.joblib
├── src/
│   ├── config.py            # single source of truth for MLflow URIs/names
│   ├── data_cleaning.py     # feature selection, imputation, split
│   ├── model_dev.py         # model strategies (linear, random forest)
│   └── evaluation.py        # regression metrics
├── steps/                   # ZenML steps (thin wrappers around src/)
├── pipelines/
│   └── training_pipeline.py # ingest → clean → train → evaluate
├── run_pipeline.py          # CLI entry point
└── app.py                   # Streamlit UI, loads models:/…/latest
```
