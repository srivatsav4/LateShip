import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALLOWED_FEATURES,
    FORBIDDEN_COLUMNS,
    ID_COLUMNS,
    TARGET,
    LeakageError,
    find_forbidden,
    validate_feature_list,  # fmt: skip
)
from src.features import compute_seller_history, seller_history_naive

HIST_COLS = ["seller_hist_order_count", "seller_hist_late_rate", "seller_orders_last_30d"]


# ---------------------------------------------------------------- forbidden columns
@pytest.mark.parametrize("col", sorted(FORBIDDEN_COLUMNS))
def test_forbidden_columns_rejected(col):
    with pytest.raises(LeakageError):
        validate_feature_list(ALLOWED_FEATURES + [col])


def test_allowed_features_are_clean():
    assert find_forbidden(ALLOWED_FEATURES) == []
    assert validate_feature_list(ALLOWED_FEATURES) == ALLOWED_FEATURES


def test_unlisted_column_rejected():
    with pytest.raises(LeakageError):
        validate_feature_list(["freight_value", "mystery_column"])


def test_actual_delivery_information_never_in_feature_table(feature_df):
    feats = [c for c in feature_df.columns if c not in ID_COLUMNS + [TARGET]]
    assert set(feats) == set(ALLOWED_FEATURES)
    assert not any(re.search("deliver(ed)?_(ts|date|customer)|review|carrier|status", c) for c in feats)


def test_model_and_schema_use_only_allowed_features(trained):
    assert set(trained["metadata"]["features"]) <= set(ALLOWED_FEATURES)


# ---------------------------------------------------------------- point-in-time seller history
def test_vectorised_history_matches_naive_reference(base):
    sub = base[base["seller_id"].isin(base["seller_id"].unique()[:4])].reset_index(drop=True)
    fast = compute_seller_history(sub).set_index("order_id")
    slow = seller_history_naive(sub).set_index("order_id").loc[fast.index]
    pd.testing.assert_frame_equal(fast[HIST_COLS], slow[HIST_COLS], check_dtype=False, check_exact=False)


def test_history_never_uses_future_orders(base):
    """Brute force: every order counted in the history was purchased AND delivered before t."""
    sub = base[base["seller_id"] == base["seller_id"].value_counts().index[0]].reset_index(drop=True)
    sub = sub.sort_values("purchase_ts").reset_index(drop=True).iloc[:300]
    hist = compute_seller_history(sub).set_index("order_id")
    for r in sub.iloc[::15].itertuples():
        allowed = sub[sub["delivered_ts"].notna() & (sub["delivered_ts"] < r.purchase_ts)
                      & (sub["purchase_ts"] < r.purchase_ts)]  # fmt: skip
        assert hist.loc[r.order_id, "seller_hist_order_count"] == len(allowed)
        assert (allowed["purchase_ts"] < r.purchase_ts).all()


def test_changing_current_and_future_outcomes_does_not_change_history(base):
    """Perturbation test: flip outcomes of order i and every later order -> features of i are unchanged."""
    seller = base["seller_id"].value_counts().index[0]
    sub = base[base["seller_id"] == seller].sort_values("purchase_ts").reset_index(drop=True)
    labelled = sub.index[sub[TARGET].notna()]
    i = labelled[len(labelled) // 2]
    t_i = sub.loc[i, "purchase_ts"]
    before = compute_seller_history(sub).set_index("order_id").loc[sub.loc[i, "order_id"], HIST_COLS]

    pert = sub.copy()
    later = pert["purchase_ts"] >= t_i  # includes the order itself
    pert.loc[later & pert[TARGET].notna(), TARGET] = 1 - pert.loc[later & pert[TARGET].notna(), TARGET]
    pert.loc[later & pert["delivered_ts"].notna(), "delivered_ts"] += pd.Timedelta(days=90)
    after = compute_seller_history(pert).set_index("order_id").loc[sub.loc[i, "order_id"], HIST_COLS]
    pd.testing.assert_series_equal(before, after)


def test_current_order_does_not_contribute_to_its_own_history():
    t = pd.Timestamp("2018-01-10 12:00")
    base = pd.DataFrame({
        "order_id": ["a", "b"], "seller_id": ["s", "s"],
        "purchase_ts": [t, t + pd.Timedelta(days=30)],
        "delivered_ts": [t + pd.Timedelta(days=2), t + pd.Timedelta(days=33)],
        TARGET: [1.0, 0.0],
    })  # fmt: skip
    h = compute_seller_history(base).set_index("order_id")
    assert h.loc["a", "seller_hist_order_count"] == 0 and np.isnan(h.loc["a", "seller_hist_late_rate"])
    assert h.loc["b", "seller_hist_order_count"] == 1 and h.loc["b", "seller_hist_late_rate"] == 1.0


def test_order_delivered_after_purchase_of_current_order_is_excluded():
    """Order A purchased earlier but delivered AFTER B was placed: unknown to B at prediction time."""
    t = pd.Timestamp("2018-03-01")
    base = pd.DataFrame({
        "order_id": ["a", "b"], "seller_id": ["s", "s"],
        "purchase_ts": [t, t + pd.Timedelta(days=5)],
        "delivered_ts": [t + pd.Timedelta(days=20), t + pd.Timedelta(days=25)],
        TARGET: [1.0, 0.0],
    })  # fmt: skip
    h = compute_seller_history(base).set_index("order_id")
    assert h.loc["b", "seller_hist_order_count"] == 0
    assert h.loc["b", "seller_orders_last_30d"] == 1  # but A's *placement* is known (no outcome used)


def test_features_for_early_orders_unchanged_when_future_data_is_removed(raw):
    """Truncating the raw data at time T must not change any feature of orders purchased before T."""
    from src.features import build_features

    full = build_features(raw)
    T = full["purchase_ts"].quantile(0.5)
    o = raw["orders"]
    keep = o[pd.to_datetime(o["order_purchase_timestamp"]) < T]["order_id"]
    trunc_raw = dict(raw, orders=o[o["order_id"].isin(keep)],
                     order_items=raw["order_items"][raw["order_items"]["order_id"].isin(keep)])  # fmt: skip
    trunc = build_features(trunc_raw)
    common = trunc["order_id"]
    a = full[full["order_id"].isin(common)].set_index("order_id").sort_index()[ALLOWED_FEATURES]
    b = trunc.set_index("order_id").sort_index()[ALLOWED_FEATURES]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_sql_matches_python(base):
    """The portable SQL implementation returns the same point-in-time features as Python."""
    sub = base[base["seller_id"].isin(base["seller_id"].unique()[:3])].reset_index(drop=True)
    sql = Path(__file__).resolve().parents[1].joinpath("sql", "features.sql").read_text()
    query = sql.split("-- BEGIN seller_history")[1].split("-- END seller_history")[0]
    fmt = "%Y-%m-%d %H:%M:%S"
    tbl = pd.DataFrame({
        "order_id": sub["order_id"], "seller_id": sub["seller_id"],
        "purchase_ts": sub["purchase_ts"].dt.strftime(fmt),
        "purchase_ts_window_start": (sub["purchase_ts"] - pd.Timedelta(days=30)).dt.strftime(fmt),
        "delivered_ts": sub["delivered_ts"].dt.strftime(fmt),
        "late_delivery": sub[TARGET],
    })  # fmt: skip
    con = sqlite3.connect(":memory:")
    tbl.to_sql("order_base", con, index=False)
    got = pd.read_sql_query(query, con).set_index("order_id")
    want = compute_seller_history(sub).set_index("order_id").loc[got.index]
    pd.testing.assert_frame_equal(got[HIST_COLS], want[HIST_COLS], check_dtype=False, check_exact=False)
