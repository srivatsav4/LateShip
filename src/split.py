"""Chronological train / validation / test splitting. Never random."""

from __future__ import annotations

import pandas as pd

from .config import TARGET
from .utils import get_logger

log = get_logger(__name__)
TS = "purchase_ts"


def monthly_volume(df: pd.DataFrame, ts: str = TS) -> pd.Series:
    """Orders per calendar month (inspect this before choosing cutoffs)."""
    return df.groupby(df[ts].dt.to_period("M")).size()


def trim_sparse_months(df: pd.DataFrame, min_orders: int, ts: str = TS) -> tuple[pd.DataFrame, dict]:
    """Drop leading/trailing months whose volume is below ``min_orders`` (ramp-up / truncated tail)."""
    vol = monthly_volume(df, ts)
    keep = vol[vol >= min_orders]
    if keep.empty:
        log.warning("No month reaches min_orders=%s; not trimming.", min_orders)
        return df, {"trimmed": False, "first_month": str(vol.index[0]), "last_month": str(vol.index[-1])}
    lo, hi = keep.index.min(), keep.index.max()
    months = df[ts].dt.to_period("M")
    out = df[(months >= lo) & (months <= hi)].copy()
    info = {"trimmed": True, "first_month": str(lo), "last_month": str(hi),
            "rows_dropped": int(len(df) - len(out))}  # fmt: skip
    return out, info


def choose_cutoffs(
    df: pd.DataFrame, train_frac: float, val_frac: float, gap_days: int = 0,
    train_end=None, val_end=None, ts: str = TS,
) -> dict:  # fmt: skip
    """Cutoffs (day-aligned) from the empirical purchase-time distribution unless given explicitly.

    train: ts < train_end;  val: train_end+gap <= ts < val_end;  test: ts >= val_end+gap
    """
    s = df[ts]
    t_end = pd.Timestamp(train_end) if train_end else s.quantile(train_frac).normalize()
    v_end = pd.Timestamp(val_end) if val_end else s.quantile(train_frac + val_frac).normalize()
    if not (s.min() < t_end < v_end <= s.max()):
        raise ValueError(f"Degenerate cutoffs train_end={t_end}, val_end={v_end}")
    return {"train_end": t_end, "val_end": v_end, "gap_days": int(gap_days)}


def temporal_split(df: pd.DataFrame, cutoffs: dict, ts: str = TS) -> dict[str, pd.DataFrame]:
    gap = pd.Timedelta(days=cutoffs.get("gap_days", 0))
    t_end, v_end = pd.Timestamp(cutoffs["train_end"]), pd.Timestamp(cutoffs["val_end"])
    parts = {
        "train": df[df[ts] < t_end],
        "val": df[(df[ts] >= t_end + gap) & (df[ts] < v_end)],
        "test": df[df[ts] >= v_end + gap],
    }
    parts = {k: v.copy() for k, v in parts.items()}
    assert_chronological(parts, ts)
    return parts


def assert_chronological(parts: dict[str, pd.DataFrame], ts: str = TS) -> None:
    for k, v in parts.items():
        if v.empty:
            raise ValueError(f"Split '{k}' is empty; check cutoffs / data volume.")
    if not parts["train"][ts].max() < parts["val"][ts].min():
        raise ValueError("Leakage: max(train) >= min(val)")
    if not parts["val"][ts].max() < parts["test"][ts].min():
        raise ValueError("Leakage: max(val) >= min(test)")


def split_report(df_all: pd.DataFrame, parts: dict[str, pd.DataFrame], cutoffs: dict, trim: dict) -> dict:
    return {
        "cutoffs": {k: (str(v) if k != "gap_days" else v) for k, v in cutoffs.items()},
        "trim": trim,
        "monthly_volume": {str(k): int(v) for k, v in monthly_volume(df_all).items()},
        "splits": {
            k: {"rows": int(len(v)), "start": str(v[TS].min()), "end": str(v[TS].max()),
                "late_rate": float(v[TARGET].mean())}
            for k, v in parts.items()
        },
    }  # fmt: skip
