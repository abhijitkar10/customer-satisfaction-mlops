# Customer Satisfaction Prediction Pipeline

An end-to-end MLOps pipeline that predicts the review score (1–5) a customer
will leave for an e-commerce order, built on the real
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from Kaggle (~100k orders, 9 relational tables).

## Data

The Kaggle dataset ships as 9 normalized CSVs (orders, payments, customers,
order items, products, reviews, ...). [scripts/download_data.py](scripts/download_data.py)
downloads them via `kagglehub` (no API key needed), joins them, and then
**aggregates back to one row per order** (97,916 orders).

That last step matters. Joining items and payments to orders fans out — an
order with 3 items and 2 payments becomes 6 rows sharing a single review
score — which duplicates orders across the train/test split and inflates the
score. Measured, with identical features and pipeline:

| Grain | rows | R² |
|-------|------|----|
| Order-item (fanned out) | 117,329 | 0.089 ← inflated by duplicate rows |
| **One row per order** | 97,916 | **0.025** ← honest |

The model trains on the order grain, which is the grain the label lives at.

**Stack:** Python · ZenML (orchestration) · MLflow (experiment tracking + model
registry) · Scikit-learn (models) · Streamlit (serving UI)

## Architecture

```
                        ZenML training pipeline
 ┌─────────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────────┐
 │ ingest_data │──▶│ clean_data │──▶│ train_model │──▶│ evaluate_model │
 └─────────────┘   └────────────┘   └─────────────┘   └────────────────┘
   read merged       engineer +         fit model,        MSE/RMSE/R²,
   CSV (98k orders)  select 17 feats,   log params +      register model
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

Predicting the review score of a **delivered** order (95,829 orders, 17 features):

| Model             | RMSE  | R²    | Registered? |
|-------------------|-------|-------|-------------|
| Always guess the mean | 1.277 | 0.000 | — (baseline) |
| Linear regression | 1.145 | 0.196 | ✅ |
| **Random forest** | **1.124** | **0.225** | ✅ (serving) |

### What moved the needle

Delivery timing dominates: on-time orders average **4.29 stars**, late orders
average **2.27**. Adding four date-derived features (`delivery_days`,
`delay_vs_promise`, `is_late`, `approval_hours`) took R² from 0.025 to 0.225 —
a **9x improvement** — and feature importance confirms they are the top three
drivers, ahead of every price and product field.

Because those features are only known after delivery, the model is scoped to
**post-delivery triage** (flag an unhappy customer before they review), not
at-purchase prediction. Orders never delivered (~2%, avg 1.76 stars) are
dropped: a different failure mode, better served by a business rule than a model.

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
├── data/                    # order-level Olist table (built from Kaggle raw CSVs)
├── deployment/
│   └── model.joblib         # bundled fallback model for the deployed app
├── scripts/
│   ├── download_data.py     # Kaggle download + relational join
│   └── export_model.py      # registry → deployment/model.joblib
├── src/
│   ├── config.py            # single source of truth for MLflow URIs/names
│   ├── data_cleaning.py     # delivery feature engineering, imputation, split
│   ├── model_dev.py         # model strategies (linear, random forest)
│   └── evaluation.py        # regression metrics
├── steps/                   # ZenML steps (thin wrappers around src/)
├── pipelines/
│   └── training_pipeline.py # ingest → clean → train → evaluate
├── run_pipeline.py          # CLI entry point
└── app.py                   # Streamlit UI, loads models:/…/latest
```
