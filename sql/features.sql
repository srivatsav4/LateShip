-- LateShip: point-in-time seller-history features in plain SQL.
--
-- Input table `order_base` (one row per order with >= 1 item; built by src/features.py::build_order_base):
--   order_id, seller_id (primary seller), purchase_ts, purchase_ts_window_start (= purchase_ts - 30 days),
--   delivered_ts (NULL unless the order is labelled), late_delivery (1/0, NULL unless labelled)
--
-- The query below is PARITY-TESTED against the Python implementation on SQLite
-- (tests/test_leakage.py::test_sql_matches_python). It uses only portable constructs, so it should run on
-- PostgreSQL unchanged, but it has NOT been executed on PostgreSQL in this repository.
--
-- Leakage rule: for an order purchased at t, history may only use orders of the same seller that were
-- DELIVERED STRICTLY BEFORE t. An order can never see itself (its delivered_ts >= its own t).

-- BEGIN seller_history
SELECT
    o.order_id,
    (SELECT COUNT(*)
       FROM order_base h
      WHERE h.seller_id = o.seller_id
        AND h.delivered_ts IS NOT NULL
        AND h.late_delivery IS NOT NULL
        AND h.delivered_ts < o.purchase_ts
        AND h.purchase_ts  < o.purchase_ts)                      AS seller_hist_order_count,
    (SELECT AVG(h.late_delivery * 1.0)
       FROM order_base h
      WHERE h.seller_id = o.seller_id
        AND h.delivered_ts IS NOT NULL
        AND h.late_delivery IS NOT NULL
        AND h.delivered_ts < o.purchase_ts
        AND h.purchase_ts  < o.purchase_ts)                      AS seller_hist_late_rate,
    (SELECT COUNT(*)
       FROM order_base h
      WHERE h.seller_id = o.seller_id
        AND h.purchase_ts >= o.purchase_ts_window_start
        AND h.purchase_ts <  o.purchase_ts)                      AS seller_orders_last_30d
FROM order_base o
-- END seller_history

-- Reference only (PostgreSQL, untested): how `order_base` is derived from the raw Olist tables.
--   late_delivery  = CASE WHEN order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
--                         THEN (order_delivered_customer_date > order_estimated_delivery_date)::int END
--   primary seller = seller of the item with the lowest order_item_id (DISTINCT ON (order_id) ... ORDER BY order_item_id)
--   purchase_ts_window_start = order_purchase_timestamp - INTERVAL '30 days'
