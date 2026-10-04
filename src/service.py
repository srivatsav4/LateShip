"""Inference service shared by the FastAPI app and the dashboard (loads versioned artifacts)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ALLOWED_FEATURES, resolve
from .evaluate import risk_level
from .features import add_derived_features
from .utils import load_json


class PredictionService:
    def __init__(self, model, threshold: dict, schema: dict, metadata: dict):
        self.model, self.threshold_info, self.schema, self.metadata = model, threshold, schema, metadata
        self.threshold = float(threshold["operating_threshold"])
        self.cutoffs = threshold["risk_cutoffs"]
        self.actions = threshold["risk_actions"]
        self.version = metadata["model_version"]
        self._explainer = None

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> "PredictionService":
        d = Path(model_dir) if model_dir else resolve("models")
        need = ["lateship_model.pkl", "threshold.json", "feature_schema.json", "metadata.json"]
        missing = [f for f in need if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts in {d}: {missing}. Run `make demo` (or `make train`).")
        return cls(joblib.load(d / "lateship_model.pkl"), load_json(d / "threshold.json"),
                   load_json(d / "feature_schema.json"), load_json(d / "metadata.json"))  # fmt: skip

    @staticmethod
    def to_frame(records: list[dict]) -> pd.DataFrame:
        """Request dicts -> feature frame (None -> NaN; derived features computed here)."""
        df = pd.DataFrame(records)
        for c in ALLOWED_FEATURES:
            if c not in df.columns:
                df[c] = np.nan
        df = df.replace({None: np.nan})
        return add_derived_features(df)[ALLOWED_FEATURES]

    def decide(self, p: float) -> dict:
        level = risk_level(p, self.cutoffs)
        return {"late_probability": float(p), "risk_level": level, "flag": bool(p >= self.threshold),
                "threshold": self.threshold, "recommended_action": self.actions[level],
                "model_version": self.version}  # fmt: skip

    def predict(self, records: list[dict]) -> list[dict]:
        probs = self.model.predict_proba(self.to_frame(records))
        return [self.decide(float(p)) for p in probs]

    def explain(self, record: dict, top_k: int = 5) -> dict:
        if self._explainer is None:
            from .explain import Explainer

            self._explainer = Explainer(self.model)
        frame = self.to_frame([record])
        decision = self.decide(float(self.model.predict_proba(frame)[0]))
        return {**decision, **self._explainer.explain_row(frame, top_k),
                "explanation_note": "SHAP values in log-odds of the raw LightGBM score (pre-calibration)."}  # fmt: skip
