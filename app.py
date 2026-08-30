"""Streamlit app: loads the latest registered model from the MLflow
model registry and scores a delivered order for review risk."""

import joblib
import mlflow.sklearn
import pandas as pd
import streamlit as st

from src.config import BUNDLED_MODEL_PATH, REGISTERED_MODEL_NAME, configure_mlflow
from src.data_cleaning import FEATURE_COLUMNS

# Reference statistics measured on the 95,829 delivered orders in the
# training data. Shown in the UI so a prediction can be read against the
# population rather than in isolation.
TEST_RMSE = 1.12
DATASET_MEAN = 4.16
COHORTS = [
    ("On time or early", 89448, 4.29, 9),
    ("1–7 days late", 3600, 2.72, 49),
    ("8+ days late", 2781, 1.70, 79),
]

st.set_page_config(page_title="Order review risk", layout="centered")

st.markdown(
    """
    <style>
      h1, h2, h3 { font-family: Georgia, "Iowan Old Style", serif; letter-spacing: -0.01em; }
      [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-size: 2.6rem; }
      .note { color: #6b6b6b; font-size: 0.86rem; line-height: 1.5; }
      .rule { border-top: 1px solid rgba(128,128,128,0.25); margin: 1.4rem 0 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Order review risk")
st.markdown(
    '<p class="note">Scores a <em>delivered</em> Olist order for the review the customer '
    "has not written yet. Reviews arrive several days after delivery, so this is the window "
    "where support can still act.</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Prefer the MLflow registry; fall back to the bundled model so a
    deployed demo never breaks if the registry is unreachable."""
    try:
        configure_mlflow()
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")
        return model, "MLflow registry, latest version"
    except Exception:
        if BUNDLED_MODEL_PATH.exists():
            return joblib.load(BUNDLED_MODEL_PATH), "bundled fallback model"
        raise


try:
    model, model_source = load_model()
except Exception:
    st.error("No model available. Train one first: `python run_pipeline.py`")
    st.stop()

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

# --- Delivery: the three inputs that carry ~58% of the model's importance ---
st.subheader("Delivery")
dcol1, dcol2, dcol3 = st.columns(3)
with dcol1:
    delivery_days = st.number_input("Days to deliver", 0, 210, 12)
with dcol2:
    promised_days = st.number_input("Days promised", 1, 210, 24)
with dcol3:
    approval_hours = st.number_input("Hours to approve payment", 0.0, 720.0, 10.0)

# Derived exactly as in training: positive delay means the order arrived late.
delay_vs_promise = delivery_days - promised_days
is_late = 1.0 if delay_vs_promise > 0 else 0.0

if is_late:
    st.markdown(
        f'<p class="note"><strong>{delay_vs_promise} days past the promised date.</strong> '
        "Only 6.7% of orders miss their date, and those that do average 2.27 stars "
        "against 4.29 for the rest.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p class="note">{abs(delay_vs_promise)} days inside the promised date. '
        "Olist quotes conservatively — the median order lands 10 days out against a "
        "23-day promise, so arriving early is the norm, not a win.</p>",
        unsafe_allow_html=True,
    )

# --- Everything else: 13 features worth ~42% combined, so kept out of the way ---
with st.expander("Order and product details"):
    st.markdown(
        '<p class="note">These 13 features together matter less than delivery timing '
        "alone. Defaults are a typical Olist order.</p>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        payment_value = st.number_input("Payment value (R$)", 0.0, 15000.0, 120.0)
        price = st.number_input("Product price (R$)", 0.0, 10000.0, 100.0)
        freight_value = st.number_input("Freight (R$)", 0.0, 500.0, 20.0)
        payment_installments = st.slider("Installments", 1, 24, 1)
        payment_sequential = st.slider("Payment sequential", 1, 10, 1)
        n_items = st.slider("Items in order", 1, 21, 1)
        product_photos_qty = st.slider("Product photos", 0, 20, 2)
    with col2:
        product_weight_g = st.number_input("Weight (g)", 0.0, 40000.0, 800.0)
        product_length_cm = st.number_input("Length (cm)", 0.0, 120.0, 25.0)
        product_height_cm = st.number_input("Height (cm)", 0.0, 120.0, 10.0)
        product_width_cm = st.number_input("Width (cm)", 0.0, 120.0, 15.0)
        product_name_lenght = st.slider("Name length", 5, 80, 45)
        product_description_lenght = st.slider("Description length", 4, 4000, 600)

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

if st.button("Score this order", type="primary"):
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

    score = max(1.0, min(5.0, float(model.predict(features)[0])))
    delta = score - DATASET_MEAN

    left, right = st.columns([1, 1.5])
    with left:
        st.metric(
            "Predicted score",
            f"{score:.2f}",
            delta=f"{delta:+.2f} vs average order",
            delta_color="normal",
        )
    with right:
        if score < 3.0:
            st.markdown(
                "**Queue for outreach today.** Scores this low sit in the band where "
                "roughly half of orders end at 1–2 stars. Contacting the customer before "
                "they write the review is the whole point of scoring at this stage."
            )
        elif score < 4.0:
            st.markdown(
                "**Worth a look if capacity allows.** Below the 4.16 average but not in "
                "the danger band. Reach these only after the sub-3.0 queue is clear."
            )
        else:
            st.markdown(
                "**No action.** This order looks like the 59% of Olist orders that end "
                "in five stars. Spending support time here costs more than it returns."
            )

    st.markdown(
        f'<p class="note">Typical error is ±{TEST_RMSE:.2f} stars (test-set RMSE), so read '
        f"<strong>{score:.2f}</strong> as roughly {max(1.0, score - TEST_RMSE):.1f}–"
        f"{min(5.0, score + TEST_RMSE):.1f}. The model ranks orders by risk; it is not "
        "accurate enough to decide a refund on its own.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("**Where this order sits in the training data**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Delivery outcome": name,
                    "Orders": f"{n:,}",
                    "Mean score": f"{mean:.2f}",
                    "Ended at 1–2 stars": f"{bad}%",
                }
                for name, n, mean, bad in COHORTS
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="note">Random forest, 100 trees, depth 10 · trained on 95,829 delivered '
    f"orders · test RMSE {TEST_RMSE:.2f}, R² 0.225 · loaded from {model_source}. "
    "Delivery lateness and product quality drive review scores; this model sees the "
    "first and not the second, which is why it ranks rather than decides.</p>",
    unsafe_allow_html=True,
)
