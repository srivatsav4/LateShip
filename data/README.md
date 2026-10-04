# Data

Raw Olist data is **not** included and is git-ignored.

1. Download the *Brazilian E-Commerce Public Dataset by Olist* (Kaggle) and place these files in `data/raw/`:
   `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_products_dataset.csv`,
   `olist_customers_dataset.csv`, `olist_sellers_dataset.csv`, `olist_geolocation_dataset.csv`
   (optional: `product_category_name_translation.csv` to use English category names).
2. Run `make features SOURCE=olist` then `make train SOURCE=olist` and `make evaluate SOURCE=olist`.

**Demo mode:** `make demo` generates *synthetic* data in `data/demo/` with the same schema. It is clearly not Olist data and is only for testing the pipeline.
Review and payment tables are intentionally never loaded.
