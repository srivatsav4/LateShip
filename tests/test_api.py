import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "sample" / "sample_order.json").read_text())


@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(str(trained["models_dir"]))) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model/info")
    body = r.json()
    assert r.status_code == 200
    assert body["model_version"] and body["calibration_method"] and "threshold_info" in body
    assert body["feature_count"] == len(body["features"])


def test_valid_predict(client):
    r = client.post("/predict", json=SAMPLE)
    body = r.json()
    assert r.status_code == 200
    assert 0 <= body["late_probability"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["flag"] == (body["late_probability"] >= body["threshold"])
    assert body["recommended_action"] in {"NO_ACTION", "MONITOR_ORDER", "PROACTIVE_ALERT"}


def test_predict_handles_missing_optional_fields(client):
    minimal = {k: SAMPLE[k] for k in ["estimated_delivery_days", "freight_value", "product_price",
                                      "purchase_weekday", "purchase_month", "item_count", "seller_state",
                                      "customer_state"]}  # fmt: skip
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200 and 0 <= r.json()["late_probability"] <= 1


def test_prediction_depends_on_input(client):
    fast = client.post("/predict", json={**SAMPLE, "estimated_delivery_days": 60}).json()["late_probability"]
    slow = client.post("/predict", json={**SAMPLE, "estimated_delivery_days": 4}).json()["late_probability"]
    assert fast != slow  # real model output, not a constant


@pytest.mark.parametrize("patch", [
    {"estimated_delivery_days": -3}, {"freight_value": -1}, {"purchase_month": 13}, {"purchase_weekday": 7},
    {"item_count": 0}, {"seller_state": "XX"}, {"customer_state": 5}, {"product_price": "abc"},
    {"seller_hist_late_rate": 1.5}, {"estimated_delivery_days": None}, {"review_score": 5},
    {"order_delivered_customer_date": "2018-01-01"},
])  # fmt: skip
def test_invalid_predict_returns_422(client, patch):
    assert client.post("/predict", json={**SAMPLE, **patch}).status_code == 422


def test_missing_required_field_returns_422(client):
    bad = dict(SAMPLE)
    bad.pop("freight_value")
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", content="not json").status_code == 422


def test_explain(client):
    r = client.post("/explain?top_k=4", json=SAMPLE)
    body = r.json()
    assert r.status_code == 200 and len(body["contributions"]) == 4
    c = body["contributions"][0]
    assert {"feature", "value", "impact", "direction"} <= set(c)
    assert c["direction"] == ("increases risk" if c["impact"] > 0 else "decreases risk")
    impacts = [abs(x["impact"]) for x in body["contributions"]]
    assert impacts == sorted(impacts, reverse=True)
    assert 0 <= body["late_probability"] <= 1


def test_explanation_is_additive_in_log_odds(trained):
    """SHAP values + base value reproduce the LightGBM raw margin (explanations come from the real model)."""
    from src.explain import Explainer
    from src.service import PredictionService

    svc = PredictionService.load(trained["models_dir"])
    frame = svc.to_frame([SAMPLE])
    ex = Explainer(svc.model)
    sv, base, _ = ex.shap_values(frame)
    from src.modeling import encode_frame

    margin = svc.model.model.clf.predict(encode_frame(frame, svc.schema), raw_score=True)[0]
    assert abs(base + sv[0].sum() - margin) < 1e-6


def test_degraded_mode_without_model(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
        assert c.post("/predict", json=SAMPLE).status_code == 503
