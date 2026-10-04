"""Metrics, threshold selection and the final test-set evaluation (run once, never tuned on)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,  # fmt: skip
)


def _div(a, b):
    return float(a / b) if b else 0.0


def confusion(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {"tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum())}  # fmt: skip


def rank_metrics(y, p) -> dict:
    """Threshold-free metrics. PR-AUC (average precision) is the primary metric."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype="float64")
    two = len(np.unique(y)) == 2
    return {"pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
            "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "prevalence": float(y.mean())}  # fmt: skip


def metrics_at(y, p, threshold, c_fn: float = 5.0, c_fp: float = 1.0) -> dict:
    """Operating-point metrics. ``threshold=None`` means 'flag nothing'."""
    y = np.asarray(y).astype(int)
    pred = np.zeros(len(y), int) if threshold is None else (np.asarray(p) >= threshold).astype(int)
    c = confusion(y, pred)
    prec, rec = _div(c["tp"], c["tp"] + c["fp"]), _div(c["tp"], c["tp"] + c["fn"])
    return {"threshold": threshold, "precision": prec, "recall": rec,
            "f1": _div(2 * prec * rec, prec + rec), "flagged_rate": float(pred.mean()),
            "estimated_cost": c_fn * c["fn"] + c_fp * c["fp"], **c}  # fmt: skip


def recall_at_precision(y, p, target: float) -> float:
    """Best achievable recall on the PR curve subject to precision >= target (0 if unreachable)."""
    prec, rec, _ = precision_recall_curve(np.asarray(y).astype(int), p)
    ok = prec[:-1] >= target
    return float(rec[:-1][ok].max()) if ok.any() else 0.0


def threshold_for_precision(y, p, target: float):
    """Lowest threshold (=> highest recall) whose precision >= target; None if unreachable."""
    prec, _, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    ok = prec[:-1] >= target
    return float(thr[ok].min()) if ok.any() else None


def threshold_table(y, p, c_fn: float, c_fp: float, step: float = 0.01) -> pd.DataFrame:
    """Threshold analysis. Cost uses ASSUMED relative costs - not real Olist business facts."""
    rows = []
    for t in np.round(np.arange(step, 1.0, step), 6):
        m = metrics_at(y, p, t, c_fn, c_fp)
        rows.append({k: m[k] for k in ["threshold", "precision", "recall", "f1", "flagged_rate",
                                       "fp", "fn", "tp", "estimated_cost"]})  # fmt: skip
    return pd.DataFrame(rows)


def threshold_for_cost(y, p, c_fn: float, c_fp: float, step: float = 0.01):
    """Threshold minimising assumed cost on the grid (ties -> higher threshold). Plus reference costs."""
    tab = threshold_table(y, p, c_fn, c_fp, step)
    best = tab[tab["estimated_cost"] == tab["estimated_cost"].min()].iloc[-1]
    y = np.asarray(y).astype(int)
    refs = {"cost_flag_nothing": float(c_fn * y.sum()), "cost_flag_everything": float(c_fp * (1 - y).sum())}
    return float(best["threshold"]), refs


def select_operating_threshold(y_val, p_val, cfg_thr: dict) -> dict:
    """Choose the operating threshold on VALIDATION probabilities only."""
    c_fn, c_fp = cfg_thr["cost_false_negative"], cfg_thr["cost_false_positive"]
    t_cost, refs = threshold_for_cost(y_val, p_val, c_fn, c_fp, cfg_thr["grid_step"])
    t_prec = threshold_for_precision(y_val, p_val, cfg_thr["target_precision"])
    strategy, note = cfg_thr["strategy"], ""
    if strategy == "precision" and t_prec is None:
        strategy, note = "cost", "target precision unreachable on validation; fell back to cost strategy"
    chosen = t_prec if strategy == "precision" else t_cost
    return {"operating_threshold": chosen, "strategy_used": strategy, "strategy_requested": cfg_thr["strategy"],
            "note": note, "threshold_cost": t_cost, "threshold_target_precision": t_prec,
            "target_precision": cfg_thr["target_precision"],
            "assumptions": {"cost_false_negative": c_fn, "cost_false_positive": c_fp,
                            "label": "ASSUMPTIONS, not real Olist business facts"},
            "validation_reference_costs": refs}  # fmt: skip


def risk_cutoffs(operating_threshold: float, medium_fraction: float) -> dict:
    return {"medium": float(operating_threshold * medium_fraction), "high": float(operating_threshold)}


def risk_level(p: float, cutoffs: dict) -> str:
    return "HIGH" if p >= cutoffs["high"] else "MEDIUM" if p >= cutoffs["medium"] else "LOW"


def full_report(y, p, threshold, target_precision, c_fn, c_fp) -> dict:
    """Everything reported for one model on one dataset (threshold fixed beforehand)."""
    return {**rank_metrics(y, p),
            "recall_at_target_precision": recall_at_precision(y, p, target_precision),
            "at_threshold": metrics_at(y, p, threshold, c_fn, c_fp)}  # fmt: skip
