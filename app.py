"""Streamlit app: loads the latest registered model from the MLflow
model registry and predicts a customer satisfaction score for an order."""

import joblib
import mlflow.sklearn
import pandas as pd
import streamlit as st

from src.config import BUNDLED_MODEL_PATH, REGISTERED_MODEL_NAME, configure_mlflow
from src.data_cleaning import FEATURE_COLUMNS

st.set_page_config(page_title="Customer Satisfaction Predictor", page_icon="⭐")
st.title("⭐ Customer Satisfaction Predictor")
st.caption(
    "Predicts the review score (1-5) a customer is likely to leave for a "
    "delivered e-commerce order, using the latest model from the MLflow registry."
)


@st.cache_resource
def load_model():
    """Prefer the MLflow registry; fall back to the bundled model so a
    deployed demo never breaks if the registry is unreachable."""
    try:
        configure_mlflow()
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")
        return model, "MLflow registry (latest)"
    except Exception:
        if BUNDLED_MODEL_PATH.exists():
            return joblib.load(BUNDLED_MODEL_PATH), "bundled model"
        raise


try:
    model, model_source = load_model()
    st.caption(f"Model source: {model_source}")
except Exception:
    st.error(
        "No model available. Run the training pipeline first:\n\n"
        "`python run_pipeline.py`"
    )
    st.stop()

st.subheader("Delivery")
st.caption("Delivery timing is the strongest driver: late orders average 2.3 stars vs 4.3 for on-time.")
dcol1, dcol2, dcol3 = st.columns(3)
with dcol1:
    delivery_days = st.number_input("Days to deliver", 0, 210, 12)
with dcol2:
    promised_days = st.number_input("Days promised at checkout", 1, 210, 24)
with dcol3:
    approval_hours = st.number_input("Hours to payment approval", 0.0, 720.0, 10.0)

# Derived exactly as in training: positive delay means the order arrived late.
delay_vs_promise = delivery_days - promised_days
is_late = 1.0 if delay_vs_promise > 0 else 0.0
if is_late:
    st.warning(f"⚠️ Arrived {delay_vs_promise} days LATE")
else:
    st.info(f"✅ Arrived {abs(delay_vs_promise)} days early")

st.subheader("Order details")
col1, col2 = st.columns(2)
with col1:
    payment_value = st.number_input("Payment value (R$)", 0.0, 15000.0, 120.0)
    price = st.number_input("Product price (R$)", 0.0, 10000.0, 100.0)
    freight_value = st.number_input("Freight value (R$)", 0.0, 500.0, 20.0)
    payment_installments = st.slider("Payment installments", 1, 24, 1)
    payment_sequential = st.slider("Payment sequential", 1, 10, 1)
    n_items = st.slider("Items in order", 1, 21, 1)
    product_photos_qty = st.slider("Product photos", 0, 20, 2)
with col2:
    product_weight_g = st.number_input("Product weight (g)", 0.0, 40000.0, 800.0)
    product_length_cm = st.number_input("Length (cm)", 0.0, 120.0, 25.0)
    product_height_cm = st.number_input("Height (cm)", 0.0, 120.0, 10.0)
    product_width_cm = st.number_input("Width (cm)", 0.0, 120.0, 15.0)
    product_name_lenght = st.slider("Product name length", 5, 80, 45)
    product_description_lenght = st.slider("Description length", 4, 4000, 600)

if st.button("Predict satisfaction", type="primary"):
    values = {
        "payment_sequential": payment_sequential,
        "payment_installments": payment_installments,
        "payment_value": payment_value,
        "price": price,
        "freight_value": freight_value,
        "product_name_lenght": product_name_lenght,
        "product_description_lenght": product_description_lenght,
        "product_photos_qty": product_photos_qty,
        "product_weight_g": product_weight_g,
        "product_length_cm": product_length_cm,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
        "n_items": n_items,
        "delivery_days": delivery_days,
        "delay_vs_promise": delay_vs_promise,
        "is_late": is_late,
        "approval_hours": approval_hours,
    }
    # Build the row in the exact column order training used.
    features = pd.DataFrame([[values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    score = float(model.predict(features)[0])
    score = max(1.0, min(5.0, score))
    st.metric("Predicted review score", f"{score:.2f} / 5")
    st.progress((score - 1) / 4)
    if score >= 4:
        st.success("Customer is likely to be satisfied 🙂")
    elif score >= 3:
        st.warning("Customer satisfaction is uncertain 😐")
    else:
        st.error("Customer is likely to be unsatisfied 🙁")
