import json

import joblib
import numpy as np

from src.calibrate import select_calibrator
from src.config import TARGET
from src.evaluate import (
    metrics_at,
    recall_at_precision,
    risk_level,
    threshold_for_cost,
    threshold_for_precision,  # fmt: skip
)
from src.evaluate_run import run_evaluation
from src.modeling import LGBMModel, fit_schema


def test_model_trains_and_predicts_valid_probabilities(trained, feature_df):
    rm = trained["risk_model"]
    p = rm.predict_proba(feature_df.head(200))
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1)) and np.isfinite(p).all()


def test_training_is_deterministic(feature_df, cfg):
    schema = fit_schema(feature_df, 30)
    params = {**cfg["model"]["lgbm"], "num_leaves": 15, "learning_rate": 0.08}
    a = LGBMModel(schema, params, 42).fit(feature_df, feature_df[TARGET]).predict_proba(feature_df.head(100))
    b = LGBMModel(schema, params, 42).fit(feature_df, feature_df[TARGET]).predict_proba(feature_df.head(100))
    np.testing.assert_allclose(a, b)


def test_artifacts_written_and_honest_about_missing_test_metrics(trained):
    d = trained["models_dir"]
    for f in ["lateship_model.pkl", "threshold.json", "feature_schema.json", "metadata.json", "comparison_models.pkl"]:
        assert (d / f).exists()
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["demo_data"] is True and "not evaluated" in meta["test_metrics"]["status"]
    assert meta["feature_count"] == len(meta["features"])
    assert 0 < json.loads((d / "threshold.json").read_text())["operating_threshold"] < 1


def test_baselines_and_both_class_weightings_trained(trained):
    names = set(trained["comparison"]["model"])
    assert {"baseline_always_on_time", "baseline_logreg_est_days_only", "baseline_logreg_all_features",
            "lightgbm_unweighted", "lightgbm_weighted"} <= names


def test_threshold_selected_on_validation_only(trained):
    thr = trained["threshold"]
    assert thr["assumptions"]["label"].startswith("ASSUMPTIONS")
    assert 0 < thr["operating_threshold"] < 1


def test_threshold_helpers():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.4, 1.0, 10)])
    t = threshold_for_precision(y, p, 0.8)
    assert t is not None and metrics_at(y, p, t)["precision"] >= 0.8
    assert threshold_for_precision(y, np.zeros(100), 0.8) is None
    assert recall_at_precision(y, p, 0.8) > 0
    tc, refs = threshold_for_cost(y, p, 5.0, 1.0)
    assert 0 < tc < 1 and refs["cost_flag_nothing"] == 50.0


def test_calibration_selection_runs_on_validation_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p**2).astype(int)  # miscalibrated on purpose
    name, cal, oof, table = select_calibrator(p, y)
    assert name in {"sigmoid", "isotonic"}  # beats raw p
    assert set(table["method"]) == {"none", "sigmoid", "isotonic"}
    assert np.all((cal.predict(p) >= 0) & (cal.predict(p) <= 1))


def test_risk_levels_monotone():
    cuts = {"medium": 0.1, "high": 0.2}
    assert [risk_level(x, cuts) for x in (0.05, 0.1, 0.19, 0.2, 0.9)] == ["LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]


def test_full_evaluation_pipeline(trained, cfg, feature_df):
    out = run_evaluation(cfg, feature_df, trained["models_dir"], trained["reports_dir"], source="demo", with_shap=True)
    r = trained["reports_dir"]
    for f in ["results.md", "results.json", "shap_summary.png", "drift_report.csv", "error_analysis_segments.csv"]:
        assert (r / f).exists()
    assert "DEMO" in (r / "results.md").read_text()
    meta = json.loads((trained["models_dir"] / "metadata.json").read_text())
    assert "pr_auc" in meta["test_metrics"]
    assert 0 <= out["final"]["pr_auc"] <= 1
    assert joblib.load(trained["models_dir"] / "lateship_model.pkl") is not None
