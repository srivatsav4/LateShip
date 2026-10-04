"""Deterministic synthetic fixtures. Tests never need the real Olist data."""

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import load_config  # noqa: E402
from src.features import build_features, build_order_base  # noqa: E402
from src.synthetic import generate_demo_data  # noqa: E402
from src.train import train  # noqa: E402

SMALL = {
    "split": {"min_orders_per_month": 30},
    "mlflow": {"enabled": False},
    "model": {"lgbm": {"n_estimators": 40}, "grid": {"num_leaves": [15], "learning_rate": [0.08],
                                                    "min_child_samples": [30]}, "cv_splits": 2},
}  # fmt: skip


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides=SMALL)


@pytest.fixture(scope="session")
def raw():
    return generate_demo_data(n_orders=3500, n_sellers=30, n_customers=1500, seed=7)


@pytest.fixture(scope="session")
def base(raw):
    return build_order_base(raw)


@pytest.fixture(scope="session")
def feature_df(raw):
    return build_features(raw)


@pytest.fixture(scope="session")
def trained(cfg, feature_df, tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    out = train(cfg, feature_df, root / "models", root / "reports", source="demo")
    out["models_dir"], out["reports_dir"] = root / "models", root / "reports"
    return out
