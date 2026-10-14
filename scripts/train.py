"""Train on your real data and write everything the app needs.

    python scripts/train.py --holdout A

Reads the pickles from the earlier parts, trains on every participant except the
held-out one, evaluates on that participant, then writes:
  models/model.joblib         feature pipeline + classifier (+ metrics)
  data/sample/demo_sets.csv   real sets from the held-out participant for the demo
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SENSOR_COLS  # noqa: E402
from src.rep_counter import benchmark  # noqa: E402
from src.training import train_bundle  # noqa: E402

REQUIRED = SENSOR_COLS + ["participant", "label", "category", "set"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interim/02_outliers_removed_chauvenets.pkl",
                    help="outlier-removed data used for training")
    ap.add_argument("--raw", default="data/interim/01_data_processed.pkl",
                    help="unfiltered data used for rep benchmark and demo sets (falls back to --data)")
    ap.add_argument("--holdout", default="A", help="participant held out for evaluation and the demo")
    ap.add_argument("--demo-sets", type=int, default=1, help="demo sets per exercise and category")
    ap.add_argument("--no-gridsearch", action="store_true",
                    help="skip the grid search and train a fixed random forest (quick run)")
    args = ap.parse_args()

    df = pd.read_pickle(args.data)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{args.data} is missing columns: {missing}")

    bundle = train_bundle(df, holdout=args.holdout, gridsearch=not args.no_gridsearch)

    raw = pd.read_pickle(args.raw) if Path(args.raw).exists() else df
    held = raw[raw["participant"] == args.holdout]
    _, rep_mae = benchmark(held)
    bundle["metrics"]["rep_mae"] = round(rep_mae, 2)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "model.joblib")

    demo = pd.concat(
        [
            g[g["set"].isin(list(g["set"].unique())[: args.demo_sets])]
            for _, g in held.groupby(["label", "category"], sort=False)
        ]
    )
    (ROOT / "data" / "sample").mkdir(parents=True, exist_ok=True)
    demo[REQUIRED].to_csv(ROOT / "data" / "sample" / "demo_sets.csv", index_label="timestamp")

    m = bundle["metrics"]
    print(f"Held-out participant {args.holdout}: accuracy {m['accuracy']:.1%}, rep MAE {m['rep_mae']}")
    print("Random forest parameters:", m["best_params"])
    print("Labels:", m["labels"])
    print(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]))
    print(f"Wrote models/model.joblib and {len(demo['set'].unique())} demo sets.")


if __name__ == "__main__":
    main()
