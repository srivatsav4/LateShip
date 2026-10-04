"""Offline drift analysis (PSI + KS) between a reference and a current dataset.

This is a batch comparison utility, NOT production monitoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with quantile bins fitted on the reference (NaNs excluded)."""
    ref, cur = ref.dropna().to_numpy(float), cur.dropna().to_numpy(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref, cur = ref.fillna("unknown").astype(str), cur.fillna("unknown").astype(str)
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(((c - r) * np.log(c / r)).sum())


def status(psi: float, warn: float = 0.10, alert: float = 0.25) -> str:
    if np.isnan(psi):
        return "n/a"
    return "ALERT" if psi >= alert else "WARN" if psi >= warn else "ok"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str], categorical: list[str],
                 bins: int = 10, warn: float = 0.10, alert: float = 0.25) -> pd.DataFrame:  # fmt: skip
    rows = []
    for f in numeric:
        a, b = ref[f].dropna(), cur[f].dropna()
        ks = ks_2samp(a, b) if len(a) and len(b) else None
        psi = psi_numeric(ref[f], cur[f], bins)
        rows.append({"feature": f, "type": "numeric", "psi": psi, "status": status(psi, warn, alert),
                     "ks_stat": float(ks.statistic) if ks else np.nan,
                     "ks_pvalue": float(ks.pvalue) if ks else np.nan,
                     "missing_ref": float(ref[f].isna().mean()), "missing_cur": float(cur[f].isna().mean())})  # fmt: skip
    for f in categorical:
        psi = psi_categorical(ref[f], cur[f])
        rows.append({"feature": f, "type": "categorical", "psi": psi, "status": status(psi, warn, alert),
                     "ks_stat": np.nan, "ks_pvalue": np.nan,
                     "missing_ref": float(ref[f].isna().mean()), "missing_cur": float(cur[f].isna().mean())})  # fmt: skip
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
