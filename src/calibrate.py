"""Probability calibration: sigmoid (Platt), isotonic, or none - chosen on VALIDATION data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, dtype="float64")


class SigmoidCalibrator:
    """Platt scaling on the logit of the raw probability."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(np.asarray(p, dtype="float64"))


CALIBRATORS = {"none": IdentityCalibrator, "sigmoid": SigmoidCalibrator, "isotonic": IsotonicCalibrator}


def cross_fitted(method: str, p: np.ndarray, y: np.ndarray, folds: int = 3) -> np.ndarray:
    """Out-of-fold calibrated probabilities (contiguous folds, no shuffling)."""
    oof = np.zeros(len(p))
    for tr, te in KFold(folds, shuffle=False).split(p):
        if len(np.unique(y[tr])) < 2:
            cal = IdentityCalibrator()
        else:
            cal = CALIBRATORS[method]().fit(p[tr], y[tr])
        oof[te] = cal.predict(p[te])
    return np.clip(oof, 0.0, 1.0)


def select_calibrator(p_val, y_val, methods=("sigmoid", "isotonic"), folds: int = 3):
    """Pick the method with the lowest out-of-fold Brier score on validation; refit on all of it.

    Returns (name, fitted_calibrator, oof_probabilities_of_chosen_method, comparison_table).
    """
    p_val, y_val = np.asarray(p_val, dtype="float64"), np.asarray(y_val)
    rows, oofs = [], {}
    for m in ["none", *methods]:
        oof = p_val if m == "none" else cross_fitted(m, p_val, y_val, folds)
        oofs[m] = oof
        rows.append({"method": m, "brier_oof": brier_score_loss(y_val, oof),
                     "log_loss_oof": log_loss(y_val, np.clip(oof, EPS, 1 - EPS), labels=[0, 1])})  # fmt: skip
    table = pd.DataFrame(rows)
    best = table.sort_values("brier_oof").iloc[0]["method"]
    return best, CALIBRATORS[best]().fit(p_val, y_val), oofs[best], table


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-curve data (quantile bins so rare-event bins are populated)."""
    frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_p, "observed_rate": frac})
