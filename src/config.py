"""Configuration loading plus the single source of truth for allowed / forbidden columns."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"

TARGET = "late_delivery"
ID_COLUMNS = ["order_id", "seller_id", "purchase_ts"]

# --- Allowed, purchase-time-safe features (every one is documented in docs/leakage_audit.md) ---
NUMERIC_FEATURES = [
    "estimated_delivery_days",      # promised date is shown at checkout
    "seller_customer_distance_km",  # static zip-prefix geography
    "freight_value",                # charged at checkout
    "product_price",                # order item price total, known at checkout
    "product_weight",               # grams, catalogue attribute
    "product_volume",               # cm^3, catalogue attribute
    "purchase_weekday",             # from purchase timestamp (0=Mon)
    "purchase_month",               # from purchase timestamp
    "item_count",                   # basket size at checkout
    "freight_ratio",                # freight / (price + freight)
    "seller_hist_order_count",      # point-in-time: seller orders already delivered before t
    "seller_hist_late_rate",        # point-in-time: late share among those orders (NaN if none)
    "seller_orders_last_30d",       # seller orders purchased in [t-30d, t): workload proxy
]
CATEGORICAL_FEATURES = ["product_category", "seller_state", "customer_state"]
ALLOWED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Forbidden: information that does not exist at purchase time, or is derived from the target ---
FORBIDDEN_COLUMNS = frozenset(
    {
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "order_approved_at",
        "order_status",
        "shipping_limit_date",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
        "delivered_ts",
        "actual_delivery_days",
        "delivery_days",
        "delivery_delay_days",
        "late_delivery",
        "is_late",
    }
)
FORBIDDEN_PATTERNS = (
    "review", "carrier", "delivered", "actual", "status", "approved",
    "shipping_limit", "late_delivery", "delay", "target", "future",
)  # fmt: skip

BR_STATES = frozenset(
    "AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO".split()
)


class LeakageError(ValueError):
    """Raised when a forbidden / non-allowed column would enter the model."""


def find_forbidden(columns: Iterable[str]) -> list[str]:
    bad = []
    for col in columns:
        low = str(col).lower()
        if low in FORBIDDEN_COLUMNS or any(p in low for p in FORBIDDEN_PATTERNS):
            bad.append(str(col))
    return bad


def validate_feature_list(features: Iterable[str]) -> list[str]:
    """Return the feature list if it is allowed, otherwise raise LeakageError."""
    features = list(features)
    bad = find_forbidden(features)
    if bad:
        raise LeakageError(f"Forbidden columns in feature list: {bad}")
    unknown = [f for f in features if f not in ALLOWED_FEATURES]
    if unknown:
        raise LeakageError(f"Columns not on the allow-list: {unknown}")
    return features


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("LATESHIP_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return _deep_merge(cfg, overrides or {})


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


FEATURE_DESCRIPTIONS = {
    "estimated_delivery_days": "Promised delivery date minus purchase time (days). Shown to the customer at checkout.",
    "seller_customer_distance_km": "Haversine distance between median zip-prefix coordinates of seller and customer.",
    "freight_value": "Total freight charged for the order (known at checkout).",
    "product_price": "Total item price of the order (known at checkout).",
    "product_weight": "Total product weight in grams (catalogue attribute).",
    "product_volume": "Total product volume in cm^3 (catalogue attribute).",
    "purchase_weekday": "Weekday of the purchase timestamp (0=Mon).",
    "purchase_month": "Month of the purchase timestamp.",
    "item_count": "Number of items in the order.",
    "freight_ratio": "freight_value / (product_price + freight_value).",
    "seller_hist_order_count": "Point-in-time: orders of the primary seller delivered strictly before purchase time.",
    "seller_hist_late_rate": "Point-in-time: late share among those orders (missing if the seller has none).",
    "seller_orders_last_30d": "Orders of the primary seller purchased in the 30 days before this order (workload proxy).",
    "product_category": "Category of the first item (rare levels -> 'other', missing -> 'unknown').",
    "seller_state": "State of the primary seller.",
    "customer_state": "State of the customer.",
}
