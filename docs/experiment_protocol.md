# Experiment protocol

1. **Build features** (`make features`): point-in-time table of delivered orders; audit counts in `reports/feature_audit.json`.
2. **Inspect volume by month** (`reports/split_report.json -> monthly_volume`). Leading/trailing months below `split.min_orders_per_month` are trimmed (ramp-up / truncated tail). Cutoffs are then inferred from the empirical purchase-time quantiles (70% / 15% / 15%), snapped to midnight, unless `split.train_end` / `split.val_end` are set. `max(train) < min(val)` and `max(val) < min(test)` are asserted on every run.
3. **Class balance** is computed from the data (`reports/class_imbalance.json`); `scale_pos_weight` = on-time/late in TRAIN.
4. **Baselines** (fitted on TRAIN): always on-time; logistic regression on `estimated_delivery_days`; logistic regression on all features.
5. **LightGBM**, with and without class weighting. A deliberately small grid (`model.grid`) is scored by mean PR-AUC over forward-chaining folds on TRAIN. The model is then refit on all TRAIN with the best parameters. Seed fixed, single thread, `deterministic=True`.
6. **Model choice:** the weighted/unweighted variant with the higher validation PR-AUC.
7. **Calibration:** `none`, sigmoid and isotonic are compared by out-of-fold Brier score (3 contiguous folds) on VALIDATION; the winner is refit on all of VALIDATION.
8. **Threshold:** chosen on out-of-fold calibrated VALIDATION probabilities. Policies: (a) minimise assumed cost `c_FN*FN + c_FP*FP` (assumption: `5:1`, configurable), (b) lowest threshold meeting a target precision. `threshold.strategy` picks the operating policy; if the target precision is unreachable it falls back to the cost policy and records that. Full threshold table: `reports/threshold_analysis_validation.csv`.
9. **Risk levels:** HIGH if `p >= threshold`; MEDIUM if `p >= medium_fraction * threshold`; else LOW. Configurable heuristics, not standards.
10. **Final test evaluation** (`make evaluate`), once, with thresholds fixed. Reports PR-AUC (primary), ROC-AUC, Brier, precision/recall/F1 at the operating threshold, confusion matrix, recall at the target precision, assumed cost.
11. **Error analysis** and **offline drift** (PSI/KS train -> test) are produced from the same predictions.

**Why accuracy is not used.** With a minority late class, "always on-time" attains high accuracy while catching no late orders. PR-AUC and recall at fixed precision reflect the ranking quality and the operational trade-off.

**Why no SMOTE.** Synthetic minority interpolation would create unrealistic rows (e.g. seller-history values inconsistent with the other columns), mixes poorly with categorical features, and distorts the base rate that calibration and cost-based thresholds depend on. Class weighting is compared instead, and calibration repairs the probabilities.
