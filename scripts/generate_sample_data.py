"""Generate deterministic SYNTHETIC DEMO data in the Olist schema -> data/demo/ (not real data)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve  # noqa: E402
from src.synthetic import write_demo_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-orders", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    demo = dict(cfg["data"]["demo"])
    if a.n_orders:
        demo["n_orders"] = a.n_orders
    out = Path(a.out) if a.out else resolve(cfg["data"]["demo_dir"])
    write_demo_data(out, seed=cfg["seed"], **demo)
    print(f"DEMO DATA (synthetic, not Olist) written to {out}")


if __name__ == "__main__":
    main()
