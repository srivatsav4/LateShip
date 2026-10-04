import numpy as np
import pandas as pd

from src.config import ALLOWED_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.features import aggregate_geolocation, haversine_km
from src.modeling import encode_frame, fit_schema, normalize_categoricals


def test_haversine_known_distances():
    # Sao Paulo -> Rio de Janeiro is ~357 km; London -> Paris ~344 km
    assert abs(float(haversine_km(-23.5505, -46.6333, -22.9068, -43.1729)) - 357) < 5
    assert abs(float(haversine_km(51.5074, -0.1278, 48.8566, 2.3522)) - 344) < 3
    assert float(haversine_km(10, 20, 10, 20)) == 0.0


def test_haversine_non_negative_symmetric_and_nan_safe():
    rng = np.random.default_rng(0)
    la1, lo1 = rng.uniform(-34, 5, 500), rng.uniform(-73, -35, 500)
    la2, lo2 = rng.uniform(-34, 5, 500), rng.uniform(-73, -35, 500)
    d = haversine_km(la1, lo1, la2, lo2)
    assert (d >= 0).all()
    np.testing.assert_allclose(d, haversine_km(la2, lo2, la1, lo1))
    assert np.isnan(haversine_km(np.nan, 0, 1, 1))


def test_geolocation_aggregation_is_median_and_robust():
    geo = pd.DataFrame({"geolocation_zip_code_prefix": [1] * 5,
                        "geolocation_lat": [-23.5, -23.51, -23.49, -23.5, -3.0],  # last row: outlier
                        "geolocation_lng": [-46.6, -46.61, -46.59, -46.6, -3.0]})  # fmt: skip
    agg = aggregate_geolocation(geo)
    assert len(agg) == 1 and abs(agg["lat"].iloc[0] + 23.5) < 0.02


def test_geolocation_drops_out_of_brazil_points():
    geo = pd.DataFrame({"geolocation_zip_code_prefix": [1, 1], "geolocation_lat": [-23.5, 40.0],
                        "geolocation_lng": [-46.6, 10.0]})  # fmt: skip
    assert aggregate_geolocation(geo)["lat"].iloc[0] == -23.5


def test_feature_schema_and_target(feature_df):
    for c in ALLOWED_FEATURES + [TARGET, "order_id", "purchase_ts"]:
        assert c in feature_df.columns
    assert set(feature_df[TARGET].unique()) <= {0, 1}
    assert (feature_df["estimated_delivery_days"] > 0).all()
    assert (feature_df["seller_customer_distance_km"].dropna() >= 0).all()
    assert feature_df["purchase_ts"].is_monotonic_increasing


def test_only_delivered_orders_are_labelled(raw, feature_df):
    delivered = raw["orders"].query("order_status == 'delivered'")["order_id"]
    assert set(feature_df["order_id"]) <= set(delivered)
    undelivered = raw["orders"].query("order_status != 'delivered'")["order_id"]
    assert not set(feature_df["order_id"]) & set(undelivered)  # never mislabelled as on-time


def test_target_matches_definition(raw, feature_df):
    o = raw["orders"].set_index("order_id").loc[feature_df["order_id"]]
    late = (pd.to_datetime(o["order_delivered_customer_date"]) > pd.to_datetime(o["order_estimated_delivery_date"]))
    np.testing.assert_array_equal(feature_df[TARGET].to_numpy(), late.astype(int).to_numpy())


def test_missing_value_handling(feature_df):
    schema = fit_schema(feature_df, 30)
    row = feature_df.head(3).copy()
    row.loc[:, ["product_weight", "seller_hist_late_rate", "seller_customer_distance_km"]] = np.nan
    row.loc[:, "product_category"] = np.nan
    row.loc[row.index[0], "seller_state"] = "ZZ-not-seen"
    norm = normalize_categoricals(row, schema)
    assert (norm["product_category"] == "unknown").all()
    assert norm.loc[row.index[0], "seller_state"] == "other"
    X = encode_frame(row, schema)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert X[CATEGORICAL_FEATURES].notna().all().all()
