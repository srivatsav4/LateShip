"""Training pipeline: baselines -> LightGBM (time-aware tuning, class weighting) -> calibration ->
threshold selection -> versioned artifacts -> MLflow.

The TEST set is never touched here. Tuning uses forward-chaining CV on TRAIN; calibration and
threshold selection use VALIDATION; the test set is evaluated once in ``src.evaluate_run``.
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from .calibrate import calibration_table, select_calibrator
from .config import CATEGORICAL_FEATURES, FEATURE_DESCRIPTIONS, NUMERIC_FEATURES, TARGET, load_config, resolve
from .evaluate import (
    full_report,
    rank_metrics,
    risk_cutoffs,
    select_operating_threshold,
    threshold_for_precision,  # fmt: skip
    threshold_table,
)
from .modeling import ConstantModel, LGBMModel, LogitModel, RiskModel, fit_schema
from .split import choose_cutoffs, split_report, temporal_split, trim_sparse_months
from .utils import get_logger, save_json, set_seed

log = get_logger(__name__)


def class_balance(parts: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for k, v in parts.items():
        pos = int(v[TARGET].sum())
        out[k] = {"rows": int(len(v)), "late": pos, "on_time": int(len(v) - pos),
                  "late_rate": pos / len(v)}  # fmt: skip
    t = out["train"]
    out["scale_pos_weight_from_train"] = t["on_time"] / max(t["late"], 1)
    return out


def tune_lgbm(train: pd.DataFrame, schema: dict, cfg: dict, spw: float | None) -> tuple[dict, pd.DataFrame]:
    """Small grid, scored by mean PR-AUC over forward-chaining (expanding window) folds on TRAIN."""
    grid = cfg["model"]["grid"]
    keys = list(grid)
    seed = cfg["seed"]
    tr = train.sort_values("purchase_ts").reset_index(drop=True)
    y = tr[TARGET].to_numpy()
    rows, combos = [], []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**cfg["model"]["lgbm"], **dict(zip(keys, combo))}
        scores = []
        for a, b in TimeSeriesSplit(n_splits=cfg["model"]["cv_splits"]).split(tr):
            if y[a].sum() == 0 or y[b].sum() == 0:
                continue
            m = LGBMModel(schema, params, seed, spw).fit(tr.iloc[a], y[a])
            scores.append(average_precision_score(y[b], m.predict_proba(tr.iloc[b])))
        combos.append(params)
        rows.append({**dict(zip(keys, combo)), "cv_pr_auc": float(np.mean(scores)) if scores else -1.0})
    table = pd.DataFrame(rows)
    return combos[int(table["cv_pr_auc"].to_numpy().argmax())], table


def train(cfg: dict, df: pd.DataFrame, models_dir=None, reports_dir=None, source: str = "demo",
          use_mlflow: bool | None = None) -> dict:  # fmt: skip
    t0 = time.time()
    seed = cfg["seed"]
    set_seed(seed)
    models_dir = Path(models_dir or resolve(cfg["paths"]["models_dir"]))
    reports_dir = Path(reports_dir or resolve(cfg["paths"]["reports_dir"]))
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sp, thr_cfg = cfg["split"], cfg["threshold"]
    c_fn, c_fp, target_p = thr_cfg["cost_false_negative"], thr_cfg["cost_false_positive"], thr_cfg["target_precision"]

    # ---- temporal split (cutoffs inferred from the actual purchase-time distribution)
    d, trim = trim_sparse_months(df, sp["min_orders_per_month"])
    cutoffs = choose_cutoffs(d, sp["train_frac"], sp["val_frac"], sp["gap_days"], sp["train_end"], sp["val_end"])
    parts = temporal_split(d, cutoffs)
    train_df, val_df = parts["train"], parts["val"]
    y_tr, y_va = train_df[TARGET].to_numpy(), val_df[TARGET].to_numpy()
    srep = split_report(d, parts, cutoffs, trim)
    cb = class_balance(parts)
    spw = cb["scale_pos_weight_from_train"]
    log.info("Split sizes: %s | late rates: %s", {k: len(v) for k, v in parts.items()},
             {k: round(v['late_rate'], 4) for k, v in cb.items() if isinstance(v, dict)})  # fmt: skip

    schema = fit_schema(train_df, cfg["features"]["min_category_count"])

    # ---- baselines + LightGBM candidates (all fitted on TRAIN, scored on VALIDATION)
    lg = cfg["model"]["logreg"]
    cand: dict[str, object] = {
        "baseline_always_on_time": ConstantModel(),
        "baseline_logreg_est_days_only": LogitModel(schema, ["estimated_delivery_days"], [], "logreg_est_days",
                                                    lg["C"], lg["max_iter"], seed),
        "baseline_logreg_all_features": LogitModel(schema, NUMERIC_FEATURES, CATEGORICAL_FEATURES, "logreg_all",
                                                   lg["C"], lg["max_iter"], seed),
    }  # fmt: skip
    for name in list(cand)[1:]:
        cand[name].fit(train_df, y_tr)

    cv_tables, best_params = {}, {}
    for tag, w in [("lightgbm_unweighted", None), ("lightgbm_weighted", spw)]:
        params, cv_tables[tag] = tune_lgbm(train_df, schema, cfg, w)
        best_params[tag] = params
        cand[tag] = LGBMModel(schema, params, seed, w).fit(train_df, y_tr)

    val_scores = {k: m.predict_proba(val_df) for k, m in cand.items()}
    comp, thr_by_model = [], {}
    for k, p in val_scores.items():
        thr = None if k == "baseline_always_on_time" else threshold_for_precision(y_va, p, target_p)
        thr_by_model[k] = thr
        rep = full_report(y_va, p, thr, target_p, c_fn, c_fp)
        comp.append({"model": k, **{m: rep[m] for m in ["pr_auc", "roc_auc", "brier", "recall_at_target_precision"]},
                     **{f"@{m}": rep["at_threshold"][m] for m in ["threshold", "precision", "recall", "f1"]}})  # fmt: skip
    comp = pd.DataFrame(comp)
    comp.to_csv(reports_dir / "model_comparison_validation.csv", index=False)
    for tag, t in cv_tables.items():
        t.to_csv(reports_dir / f"lgbm_cv_results_{tag}.csv", index=False)

    # ---- model selection by validation PR-AUC among the two LightGBM variants
    lgb_rows = comp[comp["model"].str.startswith("lightgbm")].sort_values("pr_auc", ascending=False)
    chosen = lgb_rows.iloc[0]["model"]
    final: LGBMModel = cand[chosen]
    p_val_raw = val_scores[chosen]

    # ---- calibration (VALIDATION only; method chosen by out-of-fold Brier) and thresholds
    cal_name, calibrator, p_val_cal, cal_tab = select_calibrator(
        p_val_raw, y_va, cfg["calibration"]["methods"], cfg["calibration"]["cv_folds"])  # fmt: skip
    cal_tab.to_csv(reports_dir / "calibration_comparison_validation.csv", index=False)
    calibration_table(y_va, p_val_cal).to_csv(reports_dir / "calibration_curve_validation.csv", index=False)
    thr = select_operating_threshold(y_va, p_val_cal, thr_cfg)
    cuts = risk_cutoffs(thr["operating_threshold"], cfg["risk"]["medium_fraction"])
    tt = threshold_table(y_va, p_val_cal, c_fn, c_fp, thr_cfg["grid_step"])
    tt.to_csv(reports_dir / "threshold_analysis_validation.csv", index=False)

    version = cfg["project"]["model_version"]
    risk_model = RiskModel(final, calibrator, cal_name, version)
    val_report = {
        "model_selected": chosen,
        "raw": rank_metrics(y_va, p_val_raw),
        "calibrated_out_of_fold": full_report(y_va, p_val_cal, thr["operating_threshold"], target_p, c_fn, c_fp),
        "note": "Calibrated validation metrics use out-of-fold calibration to avoid in-sample optimism.",
    }
    save_json(val_report, reports_dir / "validation_report.json")
    save_json(srep, reports_dir / "split_report.json")
    save_json(cb, reports_dir / "class_imbalance.json")

    # ---- versioned artifacts
    joblib.dump(risk_model, models_dir / "lateship_model.pkl")
    joblib.dump({"models": cand, "thresholds": thr_by_model}, models_dir / "comparison_models.pkl")
    save_json({**thr, "risk_cutoffs": cuts, "risk_actions": cfg["risk"]["actions"],
               "risk_note": "Cutoffs are configurable heuristics derived from the operating threshold, "
                            "not industry standards."}, models_dir / "threshold.json")  # fmt: skip
    save_json({**schema, "descriptions": FEATURE_DESCRIPTIONS, "target": TARGET}, models_dir / "feature_schema.json")
    metadata = {
        "model_version": version, "algorithm": "LightGBM + post-hoc calibration",
        "data_source": source, "demo_data": source == "demo",
        "demo_warning": "Trained on synthetic DEMO data; not real Olist results." if source == "demo" else None,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_period": {"start": str(train_df["purchase_ts"].min()), "end": str(train_df["purchase_ts"].max())},
        "validation_period": {"start": str(val_df["purchase_ts"].min()), "end": str(val_df["purchase_ts"].max())},
        "feature_count": len(schema["numeric"]) + len(schema["categorical"]),
        "features": schema["numeric"] + list(schema["categorical"]),
        "selected_variant": chosen,
        "class_weighting": "none" if chosen == "lightgbm_unweighted" else f"scale_pos_weight={spw:.3f}",
        "hyperparameters": best_params[chosen], "calibration_method": cal_name,
        "threshold": thr["operating_threshold"], "threshold_strategy": thr["strategy_used"],
        "risk_cutoffs": cuts, "seed": seed, "split": srep["cutoffs"],
        "validation_metrics": val_report,
        "test_metrics": {"status": "not evaluated yet - generated by `make evaluate` (src/evaluate_run.py)"},
    }  # fmt: skip
    save_json(metadata, models_dir / "metadata.json")

    if use_mlflow if use_mlflow is not None else cfg["mlflow"]["enabled"]:
        _log_mlflow(cfg, comp, best_params, chosen, cal_name, thr, val_report, spw, models_dir, reports_dir)
    log.info("Trained in %.1fs | selected=%s calibration=%s threshold=%.3f (%s)", time.time() - t0, chosen,
             cal_name, thr["operating_threshold"], thr["strategy_used"])  # fmt: skip
    return {"risk_model": risk_model, "metadata": metadata, "parts": parts, "cutoffs": cutoffs,
            "threshold": thr, "comparison": comp}  # fmt: skip


def _log_mlflow(cfg, comp, best_params, chosen, cal_name, thr, val_report, spw, models_dir, reports_dir):
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name="train"):
            mlflow.log_params({"seed": cfg["seed"], "selected_model": chosen, "calibration_method": cal_name,
                               "threshold_strategy": thr["strategy_used"], "scale_pos_weight_train": spw})  # fmt: skip
            mlflow.log_metrics({"threshold": thr["operating_threshold"],
                                "val_pr_auc": val_report["raw"]["pr_auc"], "val_roc_auc": val_report["raw"]["roc_auc"],
                                "val_brier_raw": val_report["raw"]["brier"],
                                "val_brier_calibrated": val_report["calibrated_out_of_fold"]["brier"]})  # fmt: skip
            for _, r in comp.iterrows():  # one nested run per candidate model
                with mlflow.start_run(run_name=r["model"], nested=True):
                    mlflow.log_param("model_type", r["model"])
                    if r["model"] in best_params:
                        mlflow.log_params({k: v for k, v in best_params[r["model"]].items()})
                        mlflow.log_param("class_weighting", "scale_pos_weight" if r["model"].endswith("_weighted") else "none")
                    names = {"pr_auc": "pr_auc", "roc_auc": "roc_auc", "brier": "brier",
                             "recall_at_target_precision": "recall_at_target_precision",
                             "@precision": "precision_at_val_threshold", "@recall": "recall_at_val_threshold",
                             "@f1": "f1_at_val_threshold"}
                    mlflow.log_metrics({n: float(r[k]) for k, n in names.items() if pd.notna(r[k])})
            for f in ["metadata.json", "threshold.json", "feature_schema.json"]:
                mlflow.log_artifact(str(models_dir / f))
            mlflow.log_artifact(str(reports_dir / "model_comparison_validation.csv"))
    except Exception as exc:  # tracking must never break training
        log.warning("MLflow logging skipped: %s", exc)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train LateShip")
    ap.add_argument("--source", choices=["demo", "olist"], default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    source = a.source or cfg["data"]["source"]
    path = resolve(cfg["data"]["processed_dir"]) / f"features_{source}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `make features SOURCE={source}` first.")
    train(cfg, pd.read_parquet(path), source=source, use_mlflow=False if a.no_mlflow else None)


if __name__ == "__main__":
    main()
