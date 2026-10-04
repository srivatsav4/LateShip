# Leakage audit

Prediction time is `order_purchase_timestamp` (t). For every feature the question is: *would this value have existed at t?*

## Target
`late_delivery = order_delivered_customer_date > order_estimated_delivery_date`, defined **only** for orders with
`order_status == 'delivered'`, a non-null delivery date, `delivered >= purchase` and `estimated_delivery_days > 0`.
Orders that are canceled, still in transit or lack a delivery date are **excluded**, never labelled on-time.

## Feature register
| Feature | Available at t? | Justification |
|---|---|---|
| estimated_delivery_days | yes | The promised date is displayed at checkout; computed as estimated date - purchase time. |
| seller_customer_distance_km | yes | Static geography of seller and customer zip prefixes (median coordinates, Haversine). |
| freight_value | yes | Charged at checkout. |
| product_price | yes | Item prices at checkout (summed for multi-item orders). |
| product_weight, product_volume | yes | Catalogue attributes (summed over items). |
| product_category | yes | Catalogue attribute (first item). |
| purchase_weekday, purchase_month | yes | Derived from the purchase timestamp. |
| seller_state, customer_state | yes | Static addresses known at checkout. |
| item_count | yes | Basket size at checkout. |
| freight_ratio | yes | freight / (price + freight), both known at t. |
| seller_hist_order_count | yes | Point-in-time (below). |
| seller_hist_late_rate | yes | Point-in-time (below). |
| seller_orders_last_30d | yes | Count of seller orders *placed* in [t-30d, t); uses no outcome information. |

## Point-in-time seller history
For an order of seller `s` purchased at `t`:

* history set = orders of `s` with `delivered_ts < t` (strict) that are labelled;
* because `delivered_ts >= purchase_ts` for every labelled order, every history order was also purchased before `t`;
* the order itself can never be in its own history (`delivered_ts >= t_own`, and the comparison is strict);
* an order purchased earlier but **delivered after t** is excluded - its outcome was not yet known at t;
* `seller_hist_late_rate` is NaN when there is no history (not imputed with a global rate).

Implementation: `compute_seller_history` (vectorised, sorted delivery times + `searchsorted`). A slow, obviously
correct `seller_history_naive` and a SQL version (`sql/features.sql`, executed on SQLite in tests) serve as references.

## Forbidden columns
`src/config.py` holds `FORBIDDEN_COLUMNS` and `FORBIDDEN_PATTERNS`, plus an explicit **allow-list**
(`ALLOWED_FEATURES`). `validate_feature_list` runs when the feature table is built and when the schema is fitted:

* actual delivery timestamp/date, delivered customer status, `order_status`
* review score / comment / dates
* carrier handoff timestamp, `order_approved_at` (post-purchase event), `shipping_limit_date` (conservatively excluded: seller-side deadline whose availability at t is ambiguous)
* any target-derived column (`late_delivery`, delay in days, actual delivery days)
* future seller performance, future customer behaviour

## Automated checks (`tests/test_leakage.py`)
1. Every forbidden column is rejected; columns outside the allow-list are rejected.
2. The feature table contains exactly the allowed features (no delivery/review/carrier/status columns).
3. Vectorised history == naive reference == SQL.
4. Brute force: no counted history order is purchased or delivered after t.
5. **Perturbation test:** flipping outcomes/delivery dates of an order *and all later orders* leaves its features unchanged.
6. **Truncation test:** removing all raw data after time T leaves features of orders before T unchanged.
7. Edge cases: current order not in own history; order delivered after t excluded.
I also verified by mutation (temporarily making history use all of a seller's orders) that these tests fail.

## Residual risks (honest list)
* Training and evaluation are retrospective: the population is "orders that ended up delivered", a selection effect that is unavoidable for supervised labels.
* Splitting by purchase time still leaves boundary orders whose outcome became known after the cut. `split.gap_days` can purge a gap.
* `seller_state`/`customer_state` and zip-prefix geography are treated as static.
* `purchase_month` can encode seasonality that a chronological split never observes for unseen months (see drift report).
* At serving time the caller must supply seller-history values computed as of the purchase time; the API cannot verify this.
