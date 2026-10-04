"""Deterministic SYNTHETIC DEMO data in the Olist schema. NOT real Olist data.

Delivery times are simulated from latent seller quality, distance, freight and seasonality, so seller
history is genuinely (but artificially) informative. Any metric computed on this data is a DEMO
result and says nothing about real Olist performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import OLIST_FILES, haversine_km

STATES = {  # approximate state centroids (lat, lng) and sampling weights
    "SP": (-23.5, -46.6, 0.38), "RJ": (-22.9, -43.2, 0.12), "MG": (-19.9, -43.9, 0.12),
    "RS": (-30.0, -51.2, 0.06), "PR": (-25.4, -49.3, 0.07), "BA": (-12.97, -38.5, 0.05),
    "PE": (-8.05, -34.9, 0.03), "CE": (-3.7, -38.5, 0.03), "DF": (-15.8, -47.9, 0.04),
    "SC": (-27.6, -48.5, 0.05), "GO": (-16.7, -49.3, 0.03), "AM": (-3.1, -60.0, 0.02),
}  # fmt: skip
CATEGORIES = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories",
              "housewares", "watches_gifts", "telephony", "auto", "toys", "cool_stuff", "garden_tools",
              "perfumery", "baby", "electronics", "stationery"]  # fmt: skip


def _hex_ids(rng, n, prefix=""):
    return [prefix + "".join(rng.choice(list("0123456789abcdef"), 12)) for _ in range(n)]


def generate_demo_data(n_orders=12000, n_sellers=60, n_customers=5000, start="2017-01-01",
                       months=24, seed=42) -> dict[str, pd.DataFrame]:  # fmt: skip
    rng = np.random.default_rng(seed)
    st = list(STATES)
    w = np.array([STATES[s][2] for s in st])
    w = w / w.sum()

    # geography: 8 zip prefixes per state, several noisy geolocation rows per prefix
    prefixes, geo_rows = [], []
    for si, s in enumerate(st):
        for j in range(8):
            z = 10000 + si * 1000 + j * 37
            lat = STATES[s][0] + rng.normal(0, 0.4)
            lng = STATES[s][1] + rng.normal(0, 0.4)
            prefixes.append((z, s, lat, lng))
            for _ in range(int(rng.integers(3, 7))):
                geo_rows.append((z, lat + rng.normal(0, 0.01), lng + rng.normal(0, 0.01), s))
            if rng.random() < 0.3:  # an outlier row: the median must be robust to it
                geo_rows.append((z, lat + 3.0, lng + 3.0, s))
    pre = pd.DataFrame(prefixes, columns=["z", "state", "lat", "lng"])
    geolocation = pd.DataFrame(geo_rows, columns=["geolocation_zip_code_prefix", "geolocation_lat",
                                                  "geolocation_lng", "geolocation_state"])  # fmt: skip
    coords = pre.set_index("z")[["lat", "lng"]]

    def pick_prefix(states_idx):
        out = []
        for si in states_idx:
            sub = pre[pre["state"] == st[si]]
            out.append(int(sub["z"].iloc[int(rng.integers(len(sub)))]))
        return np.array(out)

    s_state = rng.choice(len(st), n_sellers, p=w)
    s_zip = pick_prefix(s_state)
    seller_ids = _hex_ids(rng, n_sellers, "s")
    sellers = pd.DataFrame({"seller_id": seller_ids, "seller_zip_code_prefix": s_zip,
                            "seller_city": "city", "seller_state": [st[i] for i in s_state]})  # fmt: skip
    seller_slow = rng.lognormal(-0.2, 0.9, n_sellers) * 2.0  # extra days of slowness
    seller_drift = rng.normal(0, 0.04, n_sellers)  # sellers get better/worse over time (days per month)
    seller_pop = rng.dirichlet(np.ones(n_sellers) * 0.7)

    c_state = rng.choice(len(st), n_customers, p=w)
    c_zip = pick_prefix(c_state)
    cust_ids = _hex_ids(rng, n_customers, "c")
    customers = pd.DataFrame({"customer_id": cust_ids, "customer_unique_id": cust_ids,
                              "customer_zip_code_prefix": c_zip, "customer_city": "city",
                              "customer_state": [st[i] for i in c_state]})  # fmt: skip

    n_prod = 300
    cats = rng.choice(CATEGORIES, n_prod)
    products = pd.DataFrame({
        "product_id": _hex_ids(rng, n_prod, "p"),
        "product_category_name": cats,
        "product_weight_g": np.round(rng.lognormal(6.5, 1.0, n_prod)),
        "product_length_cm": np.round(rng.uniform(10, 80, n_prod)),
        "product_height_cm": np.round(rng.uniform(2, 50, n_prod)),
        "product_width_cm": np.round(rng.uniform(8, 60, n_prod)),
    })  # fmt: skip
    products.loc[rng.random(n_prod) < 0.01, "product_weight_g"] = np.nan
    products.loc[rng.random(n_prod) < 0.02, "product_category_name"] = np.nan

    # order timestamps: sparse first/last month, ramp-up in between
    m_w = np.concatenate([[0.02], np.linspace(0.3, 1.0, months - 2), [0.03]])
    month_idx = rng.choice(months, n_orders, p=m_w / m_w.sum())
    month_start = pd.date_range(start, periods=months + 1, freq="MS")
    span = (month_start[month_idx + 1] - month_start[month_idx]).total_seconds().to_numpy()
    purchase = month_start[month_idx] + pd.to_timedelta((rng.random(n_orders) * span).astype(int), unit="s")
    purchase = pd.DatetimeIndex(np.sort(purchase.values))
    months_since = (purchase - purchase[0]).days.to_numpy() / 30.0

    o_seller = rng.choice(n_sellers, n_orders, p=seller_pop)
    o_cust = rng.integers(0, n_customers, n_orders)
    n_items = rng.choice([1, 2, 3], n_orders, p=[0.8, 0.15, 0.05])

    sc = coords.loc[s_zip[o_seller]].to_numpy()
    cc = coords.loc[c_zip[o_cust]].to_numpy()
    dist = haversine_km(sc[:, 0], sc[:, 1], cc[:, 0], cc[:, 1])

    price_unit = rng.lognormal(4.3, 0.8, n_orders)
    freight = np.round(6 + dist * 0.012 + 0.002 * price_unit * n_items + rng.gamma(2, 2, n_orders), 2)
    est_days = np.clip(np.round(6 + dist / 110 + rng.uniform(3, 12, n_orders)), 5, 60)
    season = np.where(np.isin(purchase.month, [11, 12]), 2.0, 0.0)
    weekend = np.where(np.isin(purchase.weekday, [4, 5]), 1.2, 0.0)
    slow = np.maximum(seller_slow[o_seller] + seller_drift[o_seller] * months_since, 0.2)
    actual_days = (3 + dist / 120 + slow + season + weekend + 0.03 * freight
                   + rng.gamma(2.0, 1.6, n_orders))  # fmt: skip
    deliv = purchase + pd.to_timedelta((actual_days * 86400).astype(int), unit="s")
    est = (purchase.normalize() + pd.to_timedelta(est_days, unit="D") + pd.Timedelta(hours=23, minutes=59))

    status = rng.choice(["delivered", "canceled", "shipped", "invoiced"], n_orders, p=[0.965, 0.02, 0.01, 0.005])
    delivered = status == "delivered"
    order_ids = _hex_ids(rng, n_orders, "o")
    orders = pd.DataFrame({
        "order_id": order_ids, "customer_id": np.array(cust_ids)[o_cust], "order_status": status,
        "order_purchase_timestamp": purchase,
        "order_approved_at": purchase + pd.Timedelta(hours=1),
        "order_delivered_carrier_date": (purchase + pd.to_timedelta(rng.integers(1, 4, n_orders), unit="D")),
        "order_delivered_customer_date": pd.Series(deliv).where(delivered).values,
        "order_estimated_delivery_date": est,
    })  # fmt: skip
    orders.loc[status == "canceled", "order_delivered_carrier_date"] = pd.NaT

    items = []
    sid = np.array(seller_ids)
    for i in range(n_orders):
        for k in range(n_items[i]):
            s = o_seller[i] if (k == 0 or rng.random() > 0.03) else rng.integers(n_sellers)
            items.append((order_ids[i], k + 1, products["product_id"].iloc[int(rng.integers(n_prod))],
                          sid[s], purchase[i] + pd.Timedelta(days=3),
                          round(float(price_unit[i] * rng.uniform(0.8, 1.2)), 2),
                          round(float(freight[i] / n_items[i]), 2)))  # fmt: skip
    order_items = pd.DataFrame(items, columns=["order_id", "order_item_id", "product_id", "seller_id",
                                               "shipping_limit_date", "price", "freight_value"])  # fmt: skip
    return {"orders": orders, "order_items": order_items, "products": products,
            "customers": customers, "sellers": sellers, "geolocation": geolocation}  # fmt: skip


def write_demo_data(directory: str | Path, **kwargs) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = generate_demo_data(**kwargs)
    for key, df in raw.items():
        df.to_csv(directory / OLIST_FILES[key], index=False)
    (directory / "DEMO_DATA_NOT_REAL.txt").write_text(
        "Synthetic DEMO data generated by src/synthetic.py. Not real Olist data.\n"
    )
    return raw
