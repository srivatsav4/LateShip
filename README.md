# LateShip

**Leakage-Safe, Calibrated Delivery Risk Intelligence Platform** - predicts, at purchase time, whether an Olist order will arrive after its promised date, and turns the calibrated probability into a risk level, a decision and an explanation.

## Why this project is different
LateShip is not just a late-delivery classifier. It is a leakage-safe delivery risk intelligence service that connects point-in-time feature engineering, temporal validation, calibrated probabilities, business-aware decisions, explainability and production-style model serving.

Differentiators: point-in-time seller history - automated leakage audit - temporal validation - probability calibration - threshold selection - cost-aware decision layer - SHAP explanations - model versioning - offline drift analysis - FastAPI service.

> **No fabricated numbers.** This README contains no performance figures. Every metric is generated into `models/metadata.json` and `reports/results.md` when you run the pipeline. Results on the bundled synthetic generator are **DEMO results**.

## Problem
Given what is known when the customer clicks "buy", estimate P(`delivered_date > estimated_date`). Late orders are a minority class, so the project uses PR-AUC and recall at a fixed precision rather than accuracy (an "always on-time" model scores high accuracy while catching nothing).

## Key design decisions
* Every feature must exist at `order_purchase_timestamp`; an allow-list and a forbidden list enforce it in code and tests.
* Seller history is **point-in-time**: only orders delivered strictly before the current purchase.
* Chronological split with cutoffs inferred from the actual volume-by-month distribution; no random splits.
* Calibration and threshold are chosen on validation; the test set is evaluated once.
* Costs behind the threshold are **assumptions** in `configs/config.yaml`, not Olist facts.

## Architecture
```
Olist CSV -> point-in-time features -> leakage validation -> temporal split -> baselines -> LightGBM
          -> calibration -> threshold selection -> MLflow -> versioned artifacts -> FastAPI (predict / explain)
```
Details and a fuller diagram: [`docs/architecture.md`](docs/architecture.md).

## Leakage prevention
See [`docs/leakage_audit.md`](docs/leakage_audit.md). Highlights: strict `delivered_ts < t` rule, an order cannot enter its own history, perturbation test (flip outcomes of the order and all later orders -> features unchanged), truncation test (drop all data after T -> earlier features unchanged), and a SQL implementation (`sql/features.sql`) that is parity-tested against the Python one on SQLite.

## Feature table
| Feature | Meaning |
|---|---|
| estimated_delivery_days | promised date - purchase time |
| seller_customer_distance_km | Haversine between median zip-prefix coordinates |
| freight_value, product_price | totals charged at checkout |
| product_weight (g), product_volume (cm3) | catalogue attributes, summed over items |
| product_category, seller_state, customer_state | categorical (rare -> `other`, missing -> `unknown`) |
| purchase_weekday, purchase_month | from the purchase timestamp |
| item_count, freight_ratio | basket size; freight / (price + freight) |
| seller_hist_order_count, seller_hist_late_rate | point-in-time seller history |
| seller_orders_last_30d | seller orders placed in the 30 days before (workload proxy, no outcomes) |

## Excluded features
Actual delivery date/timestamp, order status, review score/comments, carrier handoff, approval time, shipping-limit date, any target-derived or future information.

## Temporal split
Sparse leading/trailing months are trimmed (`split.min_orders_per_month`), then cutoffs are taken from purchase-time quantiles (70/15/15) unless fixed in config. `max(train) < min(val)` and `max(val) < min(test)` are asserted on every run and tested. Chosen cutoffs and monthly volumes: `reports/split_report.json`.

## Baselines
Always on-time; logistic regression on `estimated_delivery_days` only; logistic regression on all features. Compared on PR-AUC, ROC-AUC, Brier, precision/recall/F1 at a validation-chosen threshold, recall at target precision.

## Model
LightGBM with a small forward-chaining grid on TRAIN (`model.grid`), fixed seed, single-threaded deterministic training; parameters logged to MLflow.

## Class imbalance
Class ratios are computed from the data (`reports/class_imbalance.json`). LightGBM is trained unweighted and with `scale_pos_weight` (= on-time/late in TRAIN) and the variant with the higher validation PR-AUC is kept. SMOTE is not used: it fabricates unrealistic rows (history features, categoricals) and distorts the base rate that calibration and cost thresholds depend on.

## Calibration
`none`, sigmoid and isotonic are compared by out-of-fold Brier score on validation (`reports/calibration_comparison_validation.csv`); reliability curve on test: `reports/calibration_curve_test.png`. Calibration never sees the test set.

## Threshold selection
Operating threshold is selected on validation: minimise assumed cost (`cost_false_negative` vs `cost_false_positive`) or reach a target precision (`threshold.strategy`). Table: `reports/threshold_analysis_validation.csv`. Risk levels are configurable heuristics derived from the threshold (HIGH >= threshold, MEDIUM >= `medium_fraction` x threshold) - not industry standards. Actions: LOW -> `NO_ACTION`, MEDIUM -> `MONITOR_ORDER`, HIGH -> `PROACTIVE_ALERT` (operational interpretation only).

## SHAP
Global importance and summary plot (`reports/shap_summary.png`) plus per-order explanations from `/explain` (feature, value, impact, direction), all computed from the trained model's SHAP values in log-odds of the uncalibrated score.

## Error analysis
`reports/error_analysis_segments.csv` and the generated findings in `reports/results.md`: FP/FN by seller state, customer state, category, distance, order value and estimated window.

## Drift monitoring
Offline PSI and KS between train and test (`reports/drift_report.csv`). This is a batch analysis, not production monitoring. Note that `purchase_month` will naturally show drift under a chronological split.

## API
```bash
make api            # http://localhost:8000/docs
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @sample/sample_order.json
curl -X POST 'localhost:8000/explain?top_k=5' -H 'content-type: application/json' -d @sample/sample_order.json
```
Endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /explain`. Responses are produced by the model; shape of `/predict`:
```json
{"late_probability": "<float>", "risk_level": "LOW|MEDIUM|HIGH", "flag": "<bool>", "threshold": "<float>",
 "recommended_action": "NO_ACTION|MONITOR_ORDER|PROACTIVE_ALERT", "model_version": "lateship-1.0.0"}
```
Pydantic validates ranges, required fields, state codes (UF) and rejects unknown fields such as `review_score` (HTTP 422). Without trained artifacts the API starts "degraded" (`/predict` -> 503).

## Docker
```bash
make demo           # or `make train SOURCE=olist` - creates models/ so the image contains a model
make docker-build && make docker-run
docker compose up   # API on :8000 and an MLflow server on :5000
```
The image uses the lean `requirements-api.txt` and does not need PostgreSQL or Olist data. (The Dockerfile has not been built in the environment that generated this repository - please run `make docker-build` to confirm.)

## MLflow
Local tracking in `mlflow.db` (SQLite, git-ignored). Each training run logs a parent run plus one nested run per candidate model (type, hyperparameters, class weighting, PR-AUC, ROC-AUC, Brier, precision/recall at the validation threshold) and the chosen calibration method and threshold.
```bash
make mlflow         # = mlflow ui --backend-store-uri sqlite:///mlflow.db  -> http://localhost:5000
```

## Testing
```bash
make test           # synthetic fixtures only; no Olist data needed
```
Covers leakage (forbidden columns, perturbation, truncation, naive/SQL parity), Haversine and geolocation, target definition, missing values, chronological splits, training determinism, probabilities in [0, 1], threshold/calibration helpers, the full evaluation pipeline and every API endpoint including 422 cases.

## CI
`.github/workflows/ci.yml`: install, `ruff check`, `pytest`, demo-pipeline smoke test. No external services.

## Project structure
```
src/            config, features, split, modeling, calibrate, train, evaluate, evaluate_run, explain,
                error_analysis, monitoring, service, synthetic, api/{main,schemas}
sql/features.sql  tests/  docs/  configs/config.yaml  scripts/  sample/  app/dashboard.py
data/ models/ reports/ notebooks/   (generated content is git-ignored)
```

## Reproduction instructions
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest

# demo (synthetic data, ~1 minute)
make demo                                  # data -> features -> train -> evaluate -> sample prediction
make api

# real data: put the Olist CSVs in data/raw/ (see data/README.md)
make features SOURCE=olist && make train SOURCE=olist && make evaluate SOURCE=olist
```
Optional dashboard: `pip install streamlit && make dashboard`. Without `make`, call the scripts in `scripts/` directly.
On the first real-data run, inspect `reports/split_report.json` (monthly volume, cutoffs) and tune `split.min_orders_per_month`.

Suggested first commit: `git init && git add . && git commit -m "Initial LateShip implementation: leakage-safe features, temporal split, calibrated LightGBM, SHAP, FastAPI"`.

## Results
Generated, not hand-written: [`reports/results.md`](reports/results.md) (after `make evaluate`) and `models/metadata.json`. Demo-data results are labelled DEMO and are not Olist performance.

## Limitations
* Primary-seller simplification for multi-seller orders; zip-prefix coordinates are approximate.
* Labels exist only for delivered orders; cancellations are invisible.
* The API trusts caller-supplied seller-history values (training-serving skew risk without a feature store).
* Cost and risk thresholds are assumptions; calibration/threshold come from one validation window.
* No causal claim: SHAP shows association, and no intervention experiment exists.
* The demo data is synthetic; its metrics do not transfer.

## Future work
Rolling-origin evaluation across several cutoffs, a point-in-time feature store/online lookup, scheduled drift checks on live traffic, conformal prediction intervals, and an A/B test of proactive alerts.

## Interview talking points
* Why the seller history uses `delivered < t`, not `purchase < t` (outcome knowledge), and how the perturbation/truncation/SQL-parity tests prove it.
* Why PR-AUC and recall at fixed precision instead of accuracy; why not SMOTE.
* Calibration chosen by out-of-fold Brier on validation; thresholds from stated cost assumptions and never tuned on test.
* Training-serving consistency (`add_derived_features`, saved schema) and the serving contract for history features.
* What the drift report does and does not tell you; honest limitations above.

### Resume-ready claims (add numbers only after you have run the pipeline on real data)
* Built a leakage-safe late-delivery prediction service using purchase-time features and point-in-time seller history, evaluated with temporal holdouts.
* Compared logistic regression baselines against LightGBM using PR-AUC and recall at fixed precision under class imbalance.
* Implemented probability calibration, business-aware threshold selection, SHAP explanations, MLflow tracking, FastAPI serving, Docker packaging and automated leakage tests.
