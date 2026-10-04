# Architecture

```
   Olist CSVs  (or PostgreSQL copy)            data/raw/*.csv   |  data/demo/*.csv (synthetic DEMO)
            |
            v
 +-------------------------------+
 | Point-in-time feature         |   src/features.py   (sql/features.sql = parity-tested SQL twin)
 | engineering                   |   geolocation median agg, Haversine, seller history as of purchase time
 +-------------------------------+
            |
            v
 +-------------------------------+
 | Leakage validation            |   src/config.py allow-list + forbidden list, tests/test_leakage.py
 +-------------------------------+
            |
            v
 +-------------------------------+
 | Temporal split                |   src/split.py   trim sparse months -> cutoffs from data -> train < val < test
 +-------------------------------+
            |
            v
 +-------------------------------+
 | Baselines                     |   always on-time | LogReg(est. days) | LogReg(all features)
 +-------------------------------+
            |
            v
 +-------------------------------+
 | LightGBM                      |   small grid, forward-chaining CV on TRAIN, +/- class weighting
 +-------------------------------+
            |
            v
 +-------------------------------+
 | Calibration                   |   none | sigmoid | isotonic, chosen by out-of-fold Brier on VALIDATION
 +-------------------------------+
            |
            v
 +-------------------------------+
 | Threshold selection           |   VALIDATION only: assumed-cost or target-precision policy; risk cutoffs
 +-------------------------------+
            |
            v
 +-------------------------------+        +-----------------------------+
 | MLflow (local, sqlite)        |  <---  | Versioned artifacts         |   models/lateship_model.pkl,
 +-------------------------------+        +-----------------------------+   threshold.json, feature_schema.json,
                                                      |                       metadata.json
                                                      v
                                          +-----------------------------+
                                          | FastAPI service             |   /health /model/info /predict /explain
                                          +-----------------------------+
                                                      |
                                                      v
                          Prediction (probability + risk + action) / SHAP explanation
                          Offline evaluation: test metrics, error analysis, drift (PSI/KS)  -> reports/
```

## Data flow and responsibilities

| Stage | Module | Output |
|---|---|---|
| Features | `src/features.py`, `scripts/build_features.py` | `data/processed/features_<source>.parquet`, `reports/feature_audit.json` |
| Split | `src/split.py` | cutoffs saved in `models/metadata.json`, `reports/split_report.json` |
| Train | `src/train.py` | `models/*`, validation reports, MLflow runs |
| Evaluate | `src/evaluate_run.py` | `reports/results.md`, `results.json`, plots, error analysis, drift |
| Serve | `src/service.py`, `src/api/` | HTTP API |

## Design notes
* **Test set discipline.** `train.py` never reads the test partition's labels. `evaluate_run.py` evaluates it once with thresholds fixed beforehand.
* **One transformation path.** Derived features (`freight_ratio`) are computed by `add_derived_features`, used by both training and serving to avoid skew.
* **Serving contract.** The API receives purchase-time features, including the seller-history values. In a real deployment these would come from a point-in-time feature lookup (the SQL in `sql/features.sql` shows the logic); the caller is responsible for supplying values computed as of the purchase moment.
* **No heavy infrastructure.** No message queues, orchestrators or cloud services are used because nothing in this project's scope requires them.
