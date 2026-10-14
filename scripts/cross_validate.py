"""Leave-one-participant-out evaluation.

    python scripts/cross_validate.py              # fixed random forest, a few minutes
    python scripts/cross_validate.py --gridsearch # repeat the grid search for every fold (slow)

Each participant takes a turn as the stranger: the feature pipeline and the random
forest are fitted on the other participants only, then scored on that participant.
Reports exercise accuracy and rep-count error per participant, plus the mean and range.

This does not write any model files. Use scripts/train.py for the model the app loads.

Rep error caveat: the squat and overhead-press cutoffs in src/rep_counter.py were tuned
on all participants, so the rep MAE column is slightly optimistic.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rep_counter import benchmark  # noqa: E402
from src.training import train_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interim/02_outliers_removed_chauvenets.pkl",
                    help="outlier-removed data used for training")
    ap.add_argument("--raw", default="data/interim/01_data_processed.pkl",
                    help="unfiltered data used for the rep benchmark (falls back to --data)")
    ap.add_argument("--gridsearch", action="store_true", help="grid search inside every fold (slow)")
    args = ap.parse_args()

    df = pd.read_pickle(args.data)
    raw = pd.read_pickle(args.raw) if Path(args.raw).exists() else df
    participants = sorted(df["participant"].unique())

    rows, confusions = [], Counter()
    for p in participants:
        print(f"holding out {p} ...", flush=True)
        bundle = train_bundle(df, holdout=p, gridsearch=args.gridsearch)
        m = bundle["metrics"]
        _, rep_mae = benchmark(raw[raw["participant"] == p])
        n_sets = int(raw[(raw["participant"] == p) & (raw["label"] != "rest")]["set"].nunique())
        rows.append(
            {
                "held out": p,
                "sets": n_sets,
                "test rows": m["n_test_rows"],
                "accuracy": m["accuracy"],
                "rep MAE": rep_mae,
            }
        )
        labels = m["labels"]
        for i, true in enumerate(labels):
            for j, pred in enumerate(labels):
                if i != j and m["confusion_matrix"][i][j]:
                    confusions[(true, pred)] += m["confusion_matrix"][i][j]

    table = pd.DataFrame(rows).set_index("held out")
    shown = table.copy()
    shown["accuracy"] = (shown["accuracy"] * 100).round(1).astype(str) + "%"
    shown["rep MAE"] = shown["rep MAE"].round(2)
    print("\n" + shown.to_string())

    pooled = (table["accuracy"] * table["test rows"]).sum() / table["test rows"].sum()
    print(
        f"\naccuracy: mean {table['accuracy'].mean():.1%} across participants "
        f"(range {table['accuracy'].min():.1%} to {table['accuracy'].max():.1%}), "
        f"pooled over all rows {pooled:.1%}"
    )
    print(
        f"rep MAE:  mean {table['rep MAE'].mean():.2f} across participants "
        f"(range {table['rep MAE'].min():.2f} to {table['rep MAE'].max():.2f})"
    )
    if confusions:
        print("\nmost common mistakes (true -> predicted, rows across all folds):")
        for (true, pred), n in confusions.most_common(5):
            print(f"  {true} -> {pred}: {n}")
    else:
        print("\nno misclassified rows in any fold")


if __name__ == "__main__":
    main()
