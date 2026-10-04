"""Leakage-safe, purchase-time feature engineering for Olist orders.

Prediction time is ``order_purchase_timestamp``. Every feature below must be computable from
information that existed at that instant. Seller history is point-in-time: for an order placed
at time t it only uses orders of the same seller that were already *delivered strictly before t*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALLOWED_FEATURES, ID_COLUMNS, TARGET, validate_feature_list
from .utils import get_logger

log = get_logger(__name__)

OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}
TRANSLATION_FILE = "product_category_name_translation.csv"  # optional
ORDER_TS_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
EARTH_RADIUS_KM = 6371.0088
BRAZIL_BBOX = {"lat": (-34.0, 6.0), "lng": (-74.0, -34.0)}  # drops obviously wrong coordinates


# ----------------------------------------------------------------------------- loading
def load_olist(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load the standard Olist CSVs from ``directory`` (also used for the synthetic demo data)."""
    directory = Path(directory)
    missing = [f for f in OLIST_FILES.values() if not (directory / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {directory}: {missing}. See data/README.md for where to put the "
            "Olist CSVs, or run `make demo-data` to create synthetic DEMO data."
        )
    raw = {}
    for key, fname in OLIST_FILES.items():
        parse = ORDER_TS_COLS if key == "orders" else None
        raw[key] = pd.read_csv(directory / fname, parse_dates=parse)
    tr = directory / TRANSLATION_FILE
    if tr.exists():
        raw["translation"] = pd.read_csv(tr)
    return raw


# ----------------------------------------------------------------------------- geography
def aggregate_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """One row per zip prefix: *median* latitude/longitude (robust to the noisy duplicates).

    Rows outside a Brazil bounding box are dropped first.
    """
    g = geo.rename(
        columns={"geolocation_zip_code_prefix": "zip_prefix", "geolocation_lat": "lat",
                 "geolocation_lng": "lng"}
    )[["zip_prefix", "lat", "lng"]]  # fmt: skip
    la, lo = BRAZIL_BBOX["lat"], BRAZIL_BBOX["lng"]
    g = g[g["lat"].between(*la) & g["lng"].between(*lo)]
    return g.groupby("zip_prefix", as_index=False).agg(lat=("lat", "median"), lng=("lng", "median"))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised, always >= 0, NaN in -> NaN out)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- order table
def build_order_base(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per order that has >= 1 item. Contains static features plus outcome columns.

    Outcome columns (``delivered_ts``, ``late_delivery``, ``is_labeled``) exist ONLY so seller history
    can be computed point-in-time and so the supervised target can be defined. They never become
    model features (enforced by ``validate_feature_list``).

    Multi-item orders: item_count/freight/price/weight/volume are summed, category is taken from the
    first item and the *primary seller* is the seller of the first item (order_item_id order).
    """
    orders = raw["orders"].copy()
    for c in ORDER_TS_COLS:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    products = raw["products"].copy()
    products["volume"] = (
        products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    )
    cat = products["product_category_name"]
    if "translation" in raw:
        tmap = dict(zip(raw["translation"].iloc[:, 0], raw["translation"].iloc[:, 1]))
        cat = cat.map(tmap).fillna(cat)
    products["product_category"] = cat
    items = raw["order_items"].merge(
        products[["product_id", "product_category", "product_weight_g", "volume"]],
        on="product_id", how="left",
    ).sort_values(["order_id", "order_item_id"])  # fmt: skip

    agg = items.groupby("order_id").agg(
        item_count=("order_item_id", "size"),
        freight_value=("freight_value", "sum"),
        product_price=("price", "sum"),
        w_sum=("product_weight_g", "sum"), w_n=("product_weight_g", "count"),
        v_sum=("volume", "sum"), v_n=("volume", "count"),
    ).reset_index()  # fmt: skip
    agg["product_weight"] = agg["w_sum"].where(agg["w_n"] > 0)
    agg["product_volume"] = agg["v_sum"].where(agg["v_n"] > 0)
    first = items.drop_duplicates("order_id", keep="first")[["order_id", "seller_id", "product_category"]]
    agg = agg.drop(columns=["w_sum", "w_n", "v_sum", "v_n"]).merge(first, on="order_id")

    geo = aggregate_geolocation(raw["geolocation"])
    sellers = raw["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]].merge(
        geo.rename(columns={"zip_prefix": "seller_zip_code_prefix", "lat": "s_lat", "lng": "s_lng"}),
        on="seller_zip_code_prefix", how="left",
    )  # fmt: skip
    customers = raw["customers"][["customer_id", "customer_zip_code_prefix", "customer_state"]].merge(
        geo.rename(columns={"zip_prefix": "customer_zip_code_prefix", "lat": "c_lat", "lng": "c_lng"}),
        on="customer_zip_code_prefix", how="left",
    )  # fmt: skip

    df = (
        orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                "order_delivered_customer_date", "order_estimated_delivery_date"]]  # fmt: skip
        .merge(agg, on="order_id", how="inner")
        .merge(sellers[["seller_id", "seller_state", "s_lat", "s_lng"]], on="seller_id", how="left")
        .merge(customers[["customer_id", "customer_state", "c_lat", "c_lng"]], on="customer_id", how="left")
        .rename(columns={
            "order_purchase_timestamp": "purchase_ts",
            "order_delivered_customer_date": "delivered_ts",
            "order_estimated_delivery_date": "estimated_ts",
        })
    )  # fmt: skip
    df["seller_customer_distance_km"] = haversine_km(df["s_lat"], df["s_lng"], df["c_lat"], df["c_lng"])
    df["estimated_delivery_days"] = (df["estimated_ts"] - df["purchase_ts"]) / pd.Timedelta(days=1)
    df["purchase_weekday"] = df["purchase_ts"].dt.weekday
    df["purchase_month"] = df["purchase_ts"].dt.month

    # Target definition: only delivered orders with a valid delivery date are labelled.
    delivered = (df["order_status"] == "delivered") & df["delivered_ts"].notna()
    valid = delivered & (df["delivered_ts"] >= df["purchase_ts"]) & (df["estimated_delivery_days"] > 0)
    df["is_labeled"] = valid
    df[TARGET] = np.where(valid, (df["delivered_ts"] > df["estimated_ts"]).astype(float), np.nan)
    df["delivered_ts"] = df["delivered_ts"].where(valid)  # un-labelled orders never count as history
    return df.drop(columns=["s_lat", "s_lng", "c_lat", "c_lng", "customer_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- seller history
def _to_ns(s: pd.Series) -> np.ndarray:
    return s.astype("datetime64[ns]").to_numpy().astype("int64")


def compute_seller_history(base: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Point-in-time seller features (vectorised, O(n log n)).

    For an order with purchase time t from seller s:
      * seller_hist_order_count = #labelled orders of s with delivered_ts  <  t
        (these were necessarily purchased before t as well, since delivered_ts >= purchase_ts)
      * seller_hist_late_rate   = late share among those orders (NaN when there are none)
      * seller_orders_last_30d  = #orders of s (any status) with purchase_ts in [t - window, t)
    An order can never be in its own history: its delivered_ts >= its own t, and the test is strict.
    """
    n = len(base)
    cnt = np.zeros(n)
    rate = np.full(n, np.nan)
    recent = np.zeros(n)
    purchase = _to_ns(base["purchase_ts"])
    deliv = np.full(n, np.iinfo("int64").max)
    ok = base["delivered_ts"].notna().to_numpy() & base[TARGET].notna().to_numpy()
    deliv[ok] = _to_ns(base.loc[ok, "delivered_ts"])
    late = base[TARGET].fillna(0).to_numpy()
    win = int(pd.Timedelta(days=window_days).value)

    for _, idx in base.groupby("seller_id", sort=False).indices.items():
        idx = np.asarray(idx)
        t = purchase[idx]
        hv = idx[ok[idx]]  # labelled orders of this seller
        order = np.argsort(deliv[hv], kind="stable")
        d_sorted = deliv[hv][order]
        csum = np.concatenate([[0.0], np.cumsum(late[hv][order])])
        k = np.searchsorted(d_sorted, t, side="left")  # delivered strictly before t
        cnt[idx] = k
        with np.errstate(invalid="ignore", divide="ignore"):
            rate[idx] = np.where(k > 0, csum[k] / np.maximum(k, 1), np.nan)
        p_sorted = np.sort(t)
        recent[idx] = np.searchsorted(p_sorted, t, side="left") - np.searchsorted(p_sorted, t - win, side="left")
    return pd.DataFrame(
        {"order_id": base["order_id"].to_numpy(), "seller_hist_order_count": cnt,
         "seller_hist_late_rate": rate, "seller_orders_last_30d": recent}
    )  # fmt: skip


def seller_history_naive(base: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Slow, obviously-correct reference implementation (used by tests to cross-check)."""
    rows = []
    win = pd.Timedelta(days=window_days)
    for r in base.itertuples(index=False):
        same = base[base["seller_id"] == r.seller_id]
        hist = same[same["delivered_ts"].notna() & same[TARGET].notna()
                    & (same["delivered_ts"] < r.purchase_ts) & (same["purchase_ts"] < r.purchase_ts)]  # fmt: skip
        rec = same[(same["purchase_ts"] >= r.purchase_ts - win) & (same["purchase_ts"] < r.purchase_ts)]
        rows.append((r.order_id, len(hist), hist[TARGET].mean() if len(hist) else np.nan, len(rec)))
    return pd.DataFrame(
        rows, columns=["order_id", "seller_hist_order_count", "seller_hist_late_rate", "seller_orders_last_30d"]
    )


# ----------------------------------------------------------------------------- final table
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived purchase-time features; shared by training and serving so they cannot diverge."""
    df = df.copy()
    total = df["product_price"].astype("float64") + df["freight_value"].astype("float64")
    df["freight_ratio"] = np.where(total > 0, df["freight_value"] / total.where(total > 0, 1.0), 0.0)
    return df


def build_features(raw: dict[str, pd.DataFrame], window_days: int = 30) -> pd.DataFrame:
    """Supervised dataset: ID columns + allowed features + target, sorted by purchase time."""
    base = build_order_base(raw)
    hist = compute_seller_history(base, window_days)
    df = add_derived_features(base.merge(hist, on="order_id", how="left"))
    n_all = len(df)
    df = df[df["is_labeled"]].copy()
    cols = ID_COLUMNS + ALLOWED_FEATURES + [TARGET]
    df = df[cols].sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    validate_feature_list([c for c in df.columns if c not in ID_COLUMNS + [TARGET]])
    df.attrs["audit"] = {
        "orders_with_items": int(n_all),
        "labelled_delivered_orders": int(len(df)),
        "dropped_unlabelled": int(n_all - len(df)),
        "late_rate": float(df[TARGET].mean()) if len(df) else None,
    }
    log.info("Feature table: %s", df.attrs["audit"])
    return df
