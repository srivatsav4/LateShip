"""End-to-end DEMO: synthetic data -> features -> train -> evaluate -> sample prediction."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def step(*args):
    print("\n$ python", *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    step("scripts/generate_sample_data.py")
    step("scripts/build_features.py", "--source", "demo")
    step("scripts/train_model.py", "--source", "demo")
    step("scripts/evaluate_model.py", "--source", "demo")
    from src.service import PredictionService

    svc = PredictionService.load(ROOT / "models")
    order = json.loads((ROOT / "sample" / "sample_order.json").read_text())
    print("\nSample prediction (DEMO model):")
    print(json.dumps(svc.predict([order])[0], indent=2))
    print("\nSee reports/results.md (DEMO RESULTS) and run `make mlflow` to browse runs.")


if __name__ == "__main__":
    main()
