"""Compare rep-counter low-pass settings across every set of every exercise.

    python scripts/tune_rep_cutoffs.py

For each exercise it tries:
  fixed cutoffs        the same low-pass cutoff for every set (your current one is marked)
  adaptive cutoffs     cutoff = factor x the set's own dominant frequency, so fast sets get a
                       higher cutoff than slow ones and the app does not need to know the category

and prints the mean absolute error in reps (all sets, heavy sets, medium sets).

Picking the best setting on the same sets you score it on flatters it, so each exercise also
gets a leave-one-participant-out check: choose the best setting using the other participants,
then score it on the held-out one.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EXERCISES, SAMPLING_HZ, SENSOR_COLS  # noqa: E402
from src.features import add_magnitudes  # noqa: E402
from src.rep_counter import count_reps, rep_params, true_reps  # noqa: E402

FIXED = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
CUTOFF_RANGE = (0.25, 1.2)  # adaptive cutoffs are clipped to this


def dominant_hz(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), d=1 / SAMPLING_HZ)
    spec = np.abs(np.fft.rfft(x))
    band = (freqs >= 0.1) & (freqs <= 1.5)
    return float(freqs[band][np.argmax(spec[band])]) if band.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interim/01_data_processed.pkl")
    a = ap.parse_args()
    df = pd.read_pickle(a.data)

    for label in [e for e in EXERCISES if e in set(df["label"])]:
        column, current = rep_params(label)
        fixed = sorted(set(FIXED + [current]))
        names = [f"fixed {c:g} Hz" + ("  (current)" if c == current else "") for c in fixed]
        names += [f"adaptive {k:g} x dominant" for k in FACTORS]

        errs, cats, parts = [], [], []
        for _, g in df[df["label"] == label].groupby("set", sort=False):
            cat, true = g["category"].iloc[0], true_reps(g["category"].iloc[0])
            raw = add_magnitudes(g[SENSOR_COLS].astype(float).interpolate(limit_direction="both"))[column]
            dom = dominant_hz(raw) or current
            cutoffs = fixed + [float(np.clip(k * dom, *CUTOFF_RANGE)) for k in FACTORS]
            errs.append([abs(count_reps(g, label, cutoff=c)[0] - true) for c in cutoffs])
            cats.append(cat)
            parts.append(g["participant"].iloc[0])
        E, cats, parts = np.array(errs, dtype=float), np.array(cats), np.array(parts)

        table = pd.DataFrame(
            {
                "all": E.mean(axis=0),
                "heavy": E[cats == "heavy"].mean(axis=0) if (cats == "heavy").any() else np.nan,
                "medium": E[cats == "medium"].mean(axis=0) if (cats == "medium").any() else np.nan,
            },
            index=names,
        ).round(2)
        n_heavy, n_medium = int((cats == "heavy").sum()), int((cats == "medium").sum())
        print(f"\n== {label}: {len(E)} sets (heavy {n_heavy}, medium {n_medium}), signal {column} ==")
        print(table.to_string())

        best = int(np.argmin(E.mean(axis=0)))
        cur = fixed.index(current)
        line = f"current {E[:, cur].mean():.2f}  |  best on these sets: {names[best].strip()} {E[:, best].mean():.2f}"
        if len(set(parts)) > 1:
            lopo = []
            for p in set(parts):
                pick = int(np.argmin(E[parts != p].mean(axis=0)))
                lopo.extend(E[parts == p, pick])
            line += f"  |  leave-one-participant-out {np.mean(lopo):.2f}"
        print(line)


if __name__ == "__main__":
    main()
