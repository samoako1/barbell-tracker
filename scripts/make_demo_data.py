"""Generate SYNTHETIC demo sets and a placeholder model so the app runs end to end.

Nothing here is real exercise data. To use your own data run scripts/train.py.

    python scripts/make_demo_data.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SENSOR_COLS  # noqa: E402
from src.synthetic import make_training_df  # noqa: E402
from src.training import train_bundle  # noqa: E402

HOLDOUT = "D"


def main():
    (ROOT / "data" / "sample").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = make_training_df()
    bundle = train_bundle(df, holdout=HOLDOUT, gridsearch=False)  # quick: placeholder only
    bundle["is_placeholder"] = True
    joblib.dump(bundle, ROOT / "models" / "model.joblib")

    # One demo set per (exercise, category) from the held-out participant
    held = df[df["participant"] == HOLDOUT]
    demo = pd.concat(
        [g[g["set"] == g["set"].iloc[0]] for _, g in held.groupby(["label", "category"], sort=False)]
    )
    demo[SENSOR_COLS + ["label", "category", "participant", "set"]].to_csv(
        ROOT / "data" / "sample" / "demo_sets.csv", index_label="timestamp"
    )
    print(f"Placeholder model trained on synthetic data. Held-out accuracy: {bundle['metrics']['accuracy']:.1%}")


if __name__ == "__main__":
    main()
