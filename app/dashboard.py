"""Optional Streamlit "ML Risk Console". Run: pip install streamlit && make dashboard

Loads the trained artifacts directly (no API process needed). The core API/pipeline do not depend on it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import streamlit as st
except ImportError:  # keep the repo usable without Streamlit
    raise SystemExit("Streamlit is not installed. Run `pip install streamlit`, then `make dashboard`.")

from src.config import BR_STATES  # noqa: E402
from src.service import PredictionService  # noqa: E402

st.set_page_config(page_title="LateShip", layout="wide")
st.title("LateShip")
st.caption("Delivery Risk Intelligence")


@st.cache_resource
def get_service():
    return PredictionService.load()


try:
    svc = get_service()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

if svc.metadata.get("demo_data"):
    st.warning("This model was trained on synthetic DEMO data. Outputs are not real Olist predictions.")

states = sorted(BR_STATES)
left, right = st.columns([1, 1])
with left:
    st.subheader("Order (purchase-time information)")
    c1, c2 = st.columns(2)
    order = {
        "estimated_delivery_days": c1.number_input("Estimated delivery days", 1.0, 365.0, 18.0),
        "seller_customer_distance_km": c2.number_input("Seller-customer distance (km)", 0.0, 20000.0, 1450.0),
        "freight_value": c1.number_input("Freight value", 0.0, 10000.0, 31.5),
        "product_price": c2.number_input("Product price", 0.0, 100000.0, 129.9),
        "product_weight": c1.number_input("Weight (g)", 0.0, 100000.0, 1800.0),
        "product_volume": c2.number_input("Volume (cm3)", 0.0, 1000000.0, 24000.0),
        "item_count": int(c1.number_input("Item count", 1, 100, 1)),
        "purchase_weekday": int(c2.number_input("Weekday (0=Mon)", 0, 6, 4)),
        "purchase_month": int(c1.number_input("Month", 1, 12, 11)),
        "seller_state": c2.selectbox("Seller state", states, index=states.index("SP")),
        "customer_state": c1.selectbox("Customer state", states, index=states.index("BA")),
        "product_category": c2.text_input("Product category", "furniture_decor"),
        "seller_hist_order_count": int(c1.number_input("Seller history: orders", 0, 1_000_000, 220)),
        "seller_hist_late_rate": c2.number_input("Seller history: late rate", 0.0, 1.0, 0.14),
        "seller_orders_last_30d": int(c1.number_input("Seller orders last 30d", 0, 1_000_000, 35)),
    }
    go = st.button("Assess risk", type="primary")

with right:
    st.subheader("Assessment")
    if go:
        res = svc.explain(order, top_k=6)
        m1, m2, m3 = st.columns(3)
        m1.metric("Late probability", f"{res['late_probability']:.1%}")
        m2.metric("Risk level", res["risk_level"])
        m3.metric("Recommended action", res["recommended_action"])
        st.caption(f"Operating threshold {res['threshold']:.3f} | flag={res['flag']} | model {res['model_version']}")
        st.subheader("Top risk factors")
        st.table([{"feature": c["feature"], "value": c["value"], "impact (log-odds)": round(c["impact"], 3),
                   "direction": c["direction"]} for c in res["contributions"]])
        st.caption(res["explanation_note"])
    else:
        st.info("Enter order details and press **Assess risk**.")
st.sidebar.write(f"Model version: {svc.version}")
