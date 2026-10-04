import pandas as pd
import pytest

from src.split import assert_chronological, choose_cutoffs, monthly_volume, temporal_split, trim_sparse_months


def test_chronological_ordering(feature_df):
    d, _ = trim_sparse_months(feature_df, 30)
    parts = temporal_split(d, choose_cutoffs(d, 0.7, 0.15))
    assert parts["train"]["purchase_ts"].max() < parts["val"]["purchase_ts"].min()
    assert parts["val"]["purchase_ts"].max() < parts["test"]["purchase_ts"].min()
    assert sum(len(p) for p in parts.values()) == len(d)  # no overlap, nothing lost (gap_days=0)
    assert not set(parts["train"]["order_id"]) & set(parts["test"]["order_id"])


def test_gap_days_drop_boundary_rows(feature_df):
    d, _ = trim_sparse_months(feature_df, 30)
    parts = temporal_split(d, {**choose_cutoffs(d, 0.7, 0.15), "gap_days": 7})
    assert parts["val"]["purchase_ts"].min() - parts["train"]["purchase_ts"].max() >= pd.Timedelta(days=7)
    assert sum(len(p) for p in parts.values()) < len(d)


def test_sparse_months_are_trimmed(feature_df):
    vol = monthly_volume(feature_df)
    d, info = trim_sparse_months(feature_df, 30)
    assert info["trimmed"] and len(d) < len(feature_df)
    assert vol.iloc[0] < 30 and monthly_volume(d).iloc[0] >= 30


def test_violation_is_detected(feature_df):
    d, _ = trim_sparse_months(feature_df, 30)
    parts = temporal_split(d, choose_cutoffs(d, 0.7, 0.15))
    bad = {"train": parts["val"], "val": parts["train"], "test": parts["test"]}
    with pytest.raises(ValueError):
        assert_chronological(bad)


def test_explicit_cutoffs_respected(feature_df):
    d, _ = trim_sparse_months(feature_df, 30)
    t_end = d["purchase_ts"].quantile(0.6).normalize()
    v_end = d["purchase_ts"].quantile(0.8).normalize()
    c = choose_cutoffs(d, 0.7, 0.15, train_end=str(t_end), val_end=str(v_end))
    parts = temporal_split(d, c)
    assert parts["train"]["purchase_ts"].max() < t_end <= parts["val"]["purchase_ts"].min()
