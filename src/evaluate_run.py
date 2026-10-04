"""Final holdout evaluation (run once after training) + SHAP, error analysis, drift and reports.

Everything written to ``reports/`` is generated from actual model outputs. Demo-data runs are
labelled as DEMO. The test set is only *evaluated* here; no threshold or model choice is tuned on it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibrate import calibration_table
from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_config, resolve
from .error_analysis import generate_findings, segment_errors
from .evaluate import full_report
from .monitoring import drift_report
from .split import temporal_split, trim_sparse_months
from .utils import get_logger, load_json, save_json

log = get_logger(__name__)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) and pd.notna(v)
                 else ("-" if pd.isna(v) else str(v)) for v in r]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot_calibration(curves: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, c in curves.items():
        ax.plot(c["mean_predicted"], c["observed_rate"], marker="o", label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed late rate")
    ax.set_title("Calibration curve (test, quantile bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_evaluation(cfg: dict, df: pd.DataFrame, models_dir=None, reports_dir=None, source: str = "demo",
                   with_shap: bool = True) -> dict:  # fmt: skip
    models_dir = Path(models_dir or resolve(cfg["paths"]["models_dir"]))
    reports_dir = Path(reports_dir or resolve(cfg["paths"]["reports_dir"]))
    reports_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(models_dir / "metadata.json")
    thr = load_json(models_dir / "threshold.json")
    rm = joblib.load(models_dir / "lateship_model.pkl")
    comp = joblib.load(models_dir / "comparison_models.pkl")
    t_cfg = cfg["threshold"]
    c_fn, c_fp, target_p = t_cfg["cost_false_negative"], t_cfg["cost_false_positive"], t_cfg["target_precision"]
    op = thr["operating_threshold"]

    d, _ = trim_sparse_months(df, cfg["split"]["min_orders_per_month"])
    cuts = {**meta["split"], "gap_days": meta["split"].get("gap_days", 0)}
    parts = temporal_split(d, cuts)  # also asserts chronological ordering
    train_df, test_df = parts["train"], parts["test"]
    y = test_df[TARGET].to_numpy()

    p_raw, p_cal = rm.predict_raw(test_df), rm.predict_proba(test_df)
    final = full_report(y, p_cal, op, target_p, c_fn, c_fp)
    final["raw"] = {k: v for k, v in full_report(y, p_raw, None, target_p, c_fn, c_fp).items() if k != "at_threshold"}

    rows = []
    for name, m in comp["models"].items():
        p = m.predict_proba(test_df)
        rep = full_report(y, p, comp["thresholds"][name], target_p, c_fn, c_fp)
        rows.append({"model": name, **{k: rep[k] for k in ["pr_auc", "roc_auc", "brier", "recall_at_target_precision"]},
                     **{f"@{k}": rep["at_threshold"][k] for k in ["threshold", "precision", "recall", "f1"]}})  # fmt: skip
    a = final["at_threshold"]
    rows.append({"model": "LATESHIP_FINAL (calibrated, operating threshold)", **{k: final[k] for k in
                 ["pr_auc", "roc_auc", "brier", "recall_at_target_precision"]},
                 **{f"@{k}": a[k] for k in ["threshold", "precision", "recall", "f1"]}})  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(reports_dir / "model_comparison_test.csv", index=False)

    curves = {"raw": calibration_table(y, p_raw), "calibrated": calibration_table(y, p_cal)}
    _plot_calibration(curves, reports_dir / "calibration_curve_test.png")

    # error analysis at the operating threshold
    seg = segment_errors(test_df, p_cal, op, cfg["error_analysis"]["min_segment_size"])
    seg.to_csv(reports_dir / "error_analysis_segments.csv", index=False)
    findings = generate_findings(seg, cfg["error_analysis"]["top_k_segments"] // 3 or 3)

    # offline drift: train vs test
    dr = drift_report(train_df, test_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES, cfg["monitoring"]["psi_bins"],
                      cfg["monitoring"]["psi_warn"], cfg["monitoring"]["psi_alert"])  # fmt: skip
    dr.to_csv(reports_dir / "drift_report.csv", index=False)

    shap_top = None
    if with_shap:
        from .explain import Explainer

        ex = Explainer(rm)
        sample = test_df.sample(min(2000, len(test_df)), random_state=cfg["seed"])
        imp = ex.global_importance(sample)
        imp.to_csv(reports_dir / "shap_global_importance.csv", index=False)
        ex.summary_plot(sample, reports_dir / "shap_summary.png", seed=cfg["seed"])
        shap_top = imp.head(8)

    banner = ("> **DEMO RESULTS - synthetic data, not real Olist performance.**\n\n" if source == "demo"
              else "> Results on the Olist dataset provided by the user.\n\n")
    prev = float(np.mean(y))
    md = [f"# LateShip results ({'DEMO DATA' if source == 'demo' else 'Olist'})\n", banner,
          f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} - model `{meta['model_version']}`, "
          f"calibration `{meta['calibration_method']}`, threshold `{op:.3f}` (strategy: {thr['strategy_used']}).\n",
          f"Test rows: {len(test_df)} | late rate (prevalence): {prev:.3%}. "
          "A model that always predicts on-time would be right "
          f"{1 - prev:.1%} of the time while catching 0% of late orders - that is why accuracy is not reported.\n",
          "## Test-set model comparison\n",
          "Thresholds for baselines/variants were chosen on validation for the target precision "
          f"({target_p}); the final model uses its validation-selected operating threshold. "
          "Brier is only meaningful for calibrated probabilities.\n", md_table(table), "",
          "## Final model confusion matrix at the operating threshold (test)\n",
          f"TN={a['tn']} FP={a['fp']} FN={a['fn']} TP={a['tp']} | assumed cost = {a['estimated_cost']:.1f} "
          f"(assumes FN:FP = {c_fn}:{c_fp}; ASSUMPTION, not an Olist fact). Reference: flag nothing = "
          f"{c_fn * y.sum():.1f}, flag everything = {c_fp * (1 - y).sum():.1f}.\n",
          "## Calibration (test)\n",
          f"Brier raw: {final['raw']['brier']:.4f} | Brier calibrated: {final['brier']:.4f}. "
          "See `calibration_curve_test.png`.\n", "## Error analysis findings (generated)\n",
          *[f"- {f}" for f in findings], "", "## Drift train -> test (PSI)\n", md_table(dr[["feature", "psi", "status", "ks_stat"]].head(8))]  # fmt: skip
    if shap_top is not None:
        md += ["", "## Global SHAP importance (top 8, mean |SHAP|, log-odds)\n", md_table(shap_top)]
    (reports_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    save_json({"final": final, "comparison": rows, "findings": findings, "source": source}, reports_dir / "results.json")

    meta["test_metrics"] = {"status": "evaluated once on the held-out test set", "data_source": source,
                            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "rows": int(len(test_df)), **{k: final[k] for k in
                            ["pr_auc", "roc_auc", "brier", "recall_at_target_precision"]},
                            "at_operating_threshold": a}  # fmt: skip
    save_json(meta, models_dir / "metadata.json")
    log.info("Evaluation written to %s", reports_dir)
    return {"final": final, "table": table, "segments": seg, "drift": dr, "findings": findings}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate LateShip on the held-out test set")
    ap.add_argument("--source", choices=["demo", "olist"], default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    source = a.source or cfg["data"]["source"]
    path = resolve(cfg["data"]["processed_dir"]) / f"features_{source}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `make features SOURCE={source}` first.")
    run_evaluation(cfg, pd.read_parquet(path), source=source, with_shap=not a.no_shap)


if __name__ == "__main__":
    main()
