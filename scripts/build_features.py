"""Raw Olist (or demo) CSVs -> leakage-safe feature table (parquet) + audit JSON."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve  # noqa: E402
from src.features import build_features, load_olist  # noqa: E402
from src.split import monthly_volume  # noqa: E402
from src.utils import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["demo", "olist"], default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    source = a.source or cfg["data"]["source"]
    raw_dir = resolve(cfg["data"]["demo_dir" if source == "demo" else "raw_dir"])
    df = build_features(load_olist(raw_dir), cfg["features"]["history_window_days"])
    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"features_{source}.parquet", index=False)
    rep = resolve(cfg["paths"]["reports_dir"])
    save_json({**df.attrs["audit"], "source": source,
               "monthly_volume": {str(k): int(v) for k, v in monthly_volume(df).items()}},
              rep / "feature_audit.json")  # fmt: skip
    print(f"Wrote {len(df)} rows -> {out_dir / f'features_{source}.parquet'}")


if __name__ == "__main__":
    main()
