"""SHAP explanations computed from the real LightGBM model (never fabricated).

Explanations are in log-odds of the *raw* (uncalibrated) LightGBM score; calibration is a monotone
post-hoc mapping, so the direction/ranking of effects is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import RiskModel, encode_frame, normalize_categoricals


class Explainer:
    def __init__(self, risk_model: RiskModel):
        import shap  # lazy: slow import, only needed for explanations

        self.rm = risk_model
        self.schema = risk_model.schema
        self.features = self.schema["numeric"] + list(self.schema["categorical"])
        self._explainer = shap.TreeExplainer(risk_model.model.clf)

    def shap_values(self, df: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
        X = encode_frame(df, self.schema)[self.features]
        sv = self._explainer.shap_values(X)
        if isinstance(sv, list):  # older shap: [class0, class1]
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        base = np.atleast_1d(self._explainer.expected_value)[-1]
        return sv, float(base), X

    def global_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        sv, _, _ = self.shap_values(df)
        imp = pd.DataFrame({"feature": self.features, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def explain_row(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        """Structured explanation for a single-row frame: feature, value, impact, direction."""
        sv, base, _ = self.shap_values(row)
        shown = normalize_categoricals(row, self.schema).iloc[0]
        order = np.argsort(-np.abs(sv[0]))[:top_k]
        out = []
        for i in order:
            f = self.features[i]
            v = shown[f]
            v = None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
            impact = float(sv[0][i])
            out.append({"feature": f, "value": v, "impact": impact,
                        "direction": "increases risk" if impact > 0 else "decreases risk"})  # fmt: skip
        return {"base_value_log_odds": base, "raw_margin_log_odds": float(base + sv[0].sum()),
                "contributions": out}  # fmt: skip

    def summary_plot(self, df: pd.DataFrame, path: str | Path, max_rows: int = 2000, seed: int = 42) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        d = df.sample(min(max_rows, len(df)), random_state=seed)
        sv, _, X = self.shap_values(d)
        Xn = X.copy()
        for c in self.schema["categorical"]:
            Xn[c] = Xn[c].cat.codes  # numeric codes only for colouring
        shap.summary_plot(sv, Xn, feature_names=self.features, show=False)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
