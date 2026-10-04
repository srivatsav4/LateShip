"""Error analysis: where does the operating point produce false positives / false negatives?

All findings are generated from the evaluated predictions; nothing is hand-written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET

BUCKETS = {
    "distance_bucket": ("seller_customer_distance_km", [0, 100, 300, 600, 1000, 2000, np.inf]),
    "order_value_bucket": ("order_value", [0, 50, 100, 200, 500, np.inf]),
    "estimated_window_bucket": ("estimated_delivery_days", [0, 10, 15, 20, 25, 30, np.inf]),
}
SEGMENTS = ["seller_state", "customer_state", "product_category",
            "distance_bucket", "order_value_bucket", "estimated_window_bucket"]  # fmt: skip


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["order_value"] = d["product_price"] + d["freight_value"]
    for name, (col, edges) in BUCKETS.items():
        d[name] = pd.cut(d[col], bins=edges, right=False).astype(object).where(d[col].notna(), "missing").astype(str)
    for c in ("seller_state", "customer_state", "product_category"):
        d[c] = d[c].fillna("unknown").astype(str)
    return d


def segment_errors(df: pd.DataFrame, prob: np.ndarray, threshold: float, min_size: int = 30) -> pd.DataFrame:
    d = add_buckets(df)
    d["y"] = d[TARGET].astype(int)
    d["flag"] = (np.asarray(prob) >= threshold).astype(int)
    d["fp"] = ((d["flag"] == 1) & (d["y"] == 0)).astype(int)
    d["fn"] = ((d["flag"] == 0) & (d["y"] == 1)).astype(int)
    d["tp"] = ((d["flag"] == 1) & (d["y"] == 1)).astype(int)
    rows = []
    for seg in SEGMENTS:
        for level, g in d.groupby(seg):
            if len(g) < min_size:
                continue
            pos, neg = g["y"].sum(), (1 - g["y"]).sum()
            rows.append({"segment": seg, "level": level, "n": len(g), "late_rate": g["y"].mean(),
                         "fp": int(g["fp"].sum()), "fn": int(g["fn"].sum()), "tp": int(g["tp"].sum()),
                         "fp_rate": g["fp"].sum() / neg if neg else np.nan,
                         "fn_rate": g["fn"].sum() / pos if pos else np.nan,  # = 1 - recall
                         "precision": g["tp"].sum() / max(g["flag"].sum(), 1)})  # fmt: skip
    return pd.DataFrame(rows)


def generate_findings(seg: pd.DataFrame, top_k: int = 5, min_positives: int = 10) -> list[str]:
    """Programmatic findings: worst segments by false-negative rate and false-positive rate."""
    out = []
    if seg.empty:
        return ["No segment met the minimum size."]
    s = seg.assign(pos=(seg["late_rate"] * seg["n"]).round())
    worst_fn = s[s["pos"] >= min_positives].sort_values("fn_rate", ascending=False).head(top_k)
    for r in worst_fn.itertuples():
        out.append(f"Missed-late rate {r.fn_rate:.0%} in {r.segment}={r.level} (n={r.n}, late rate {r.late_rate:.1%}).")
    worst_fp = seg.sort_values("fp_rate", ascending=False).head(top_k)
    for r in worst_fp.itertuples():
        out.append(f"False-alarm rate {r.fp_rate:.0%} of on-time orders in {r.segment}={r.level} (n={r.n}).")
    return out
