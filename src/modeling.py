"""Model wrappers: every model takes the *raw feature frame* and handles its own encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, validate_feature_list

OTHER, UNKNOWN = "other", "unknown"


# ----------------------------------------------------------------------------- schema / encoding
def fit_schema(train: pd.DataFrame, min_count: int = 30) -> dict:
    """Feature schema learned on TRAIN only: feature lists + frequent category levels."""
    validate_feature_list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    levels = {}
    for col in CATEGORICAL_FEATURES:
        vc = train[col].fillna(UNKNOWN).astype(str).value_counts()
        keep = sorted(vc[(vc >= min_count) & (vc.index != UNKNOWN)].index.tolist())
        levels[col] = keep + [OTHER, UNKNOWN]
    return {"numeric": list(NUMERIC_FEATURES), "categorical": levels, "other_label": OTHER,
            "unknown_label": UNKNOWN}  # fmt: skip


def normalize_categoricals(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Missing -> 'unknown'; unseen/rare level -> 'other'. Returns a copy with string columns."""
    out = df.copy()
    for col, levels in schema["categorical"].items():
        s = out[col].astype(object).where(out[col].notna(), UNKNOWN).astype(str).str.strip()
        out[col] = s.where(s.isin(levels), OTHER)
    return out


def encode_frame(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """LightGBM view: numeric float columns + pandas Categoricals with fixed, saved categories."""
    norm = normalize_categoricals(df, schema)
    X = pd.DataFrame(index=norm.index)
    for col in schema["numeric"]:
        X[col] = pd.to_numeric(norm[col], errors="coerce").astype("float64")
    for col, levels in schema["categorical"].items():
        X[col] = pd.Categorical(norm[col], categories=levels)
    return X


# ----------------------------------------------------------------------------- models
class ConstantModel:
    """Baseline 1: always predict on-time (score 0 for every order)."""

    name = "always_on_time"

    def fit(self, df, y):
        return self

    def predict_proba(self, df) -> np.ndarray:
        return np.zeros(len(df))


class LogitModel:
    """Logistic regression on a chosen subset of the allowed features (baselines 2 and 3)."""

    def __init__(self, schema: dict, numeric: list[str], categorical: list[str], name: str,
                 C: float = 1.0, max_iter: int = 1000, seed: int = 42):  # fmt: skip
        validate_feature_list(numeric + categorical)
        self.schema, self.numeric, self.categorical, self.name = schema, numeric, categorical, name
        steps = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                   ("sc", StandardScaler())]), numeric)]  # fmt: skip
        if categorical:
            cats = [schema["categorical"][c] for c in categorical]
            steps.append(("cat", OneHotEncoder(categories=cats, handle_unknown="ignore"), categorical))
        self.pipe = Pipeline([
            ("pre", ColumnTransformer(steps)),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=seed)),
        ])  # fmt: skip

    def _x(self, df):
        return normalize_categoricals(df, self.schema)[self.numeric + self.categorical]

    def fit(self, df, y):
        self.pipe.fit(self._x(df), y)
        return self

    def predict_proba(self, df) -> np.ndarray:
        return self.pipe.predict_proba(self._x(df))[:, 1]


class LGBMModel:
    """Main model. ``scale_pos_weight`` implements class weighting (None = unweighted)."""

    name = "lightgbm"

    def __init__(self, schema: dict, params: dict, seed: int = 42, scale_pos_weight: float | None = None):
        self.schema, self.params, self.seed, self.spw = schema, dict(params), seed, scale_pos_weight
        kw = dict(self.params, objective="binary", random_state=seed, deterministic=True,
                  force_row_wise=True, verbose=-1)  # fmt: skip
        if scale_pos_weight:
            kw["scale_pos_weight"] = float(scale_pos_weight)
        self.clf = LGBMClassifier(**kw)

    def fit(self, df, y):
        self.clf.fit(encode_frame(df, self.schema), np.asarray(y))
        return self

    def predict_proba(self, df) -> np.ndarray:
        return self.clf.predict_proba(encode_frame(df, self.schema))[:, 1]


class RiskModel:
    """Deployable artifact: LightGBM + fitted calibrator. ``predict_proba`` returns calibrated P(late)."""

    def __init__(self, model: LGBMModel, calibrator, calibration_method: str, version: str):
        self.model, self.calibrator = model, calibrator
        self.calibration_method, self.version = calibration_method, version
        self.schema = model.schema

    def predict_raw(self, df) -> np.ndarray:
        return self.model.predict_proba(df)

    def predict_proba(self, df) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.predict_raw(df)), 0.0, 1.0)
