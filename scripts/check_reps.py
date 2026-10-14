"""Show where the rep counter finds, and misses, reps for one exercise.

    python scripts/check_reps.py --label ohp --category medium

Prints three things.

1. One row per set:
     counted vs true   the miss
     first_peak_s      a late first peak suggests a rep at the very start was missed
     end_gap_s         a large gap after the last peak suggests a rep at the very end was missed
     max_gap_s         about twice median_gap_s means a rep was missed in the middle
     start_level / end_level   where the filtered signal sits at the edges (0 = lowest, 1 = highest);
                       close to 1 means the set starts or ends at a peak, which has no neighbour on
                       one side and so is not counted as a local maximum
2. Frequency check, per set:
     reps_per_s        true reps / recording length, a rough rep frequency
     dominant_hz       strongest frequency in the raw movement signal (0.1 to 1.5 Hz)
     cutoff_hz         the low-pass cutoff used for this exercise
     If reps_per_s or dominant_hz is above cutoff_hz, the filter is smoothing the reps away.
3. Cutoff sweep: the count each set gets at other cutoffs, and the mean absolute error of each.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SAMPLING_HZ, SENSOR_COLS  # noqa: E402
from src.features import add_magnitudes  # noqa: E402
from src.rep_counter import count_reps, rep_params, true_reps  # noqa: E402

SWEEP = [0.35, 0.5, 0.65, 0.8, 1.0]


def dominant_hz(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), d=1 / SAMPLING_HZ)
    spec = np.abs(np.fft.rfft(x))
    band = (freqs >= 0.1) & (freqs <= 1.5)
    return round(float(freqs[band][np.argmax(spec[band])]), 2) if band.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interim/01_data_processed.pkl")
    ap.add_argument("--label", default="squat")
    ap.add_argument("--category", default="heavy")
    ap.add_argument("--participant", default=None, help="limit to one participant")
    a = ap.parse_args()

    df = pd.read_pickle(a.data)
    df = df[(df["label"] == a.label) & (df["category"] == a.category)]
    if a.participant:
        df = df[df["participant"] == a.participant]
    if df.empty:
        sys.exit("No sets match that label/category.")

    column, cutoff = rep_params(a.label)
    true = true_reps(a.category)
    cutoffs = sorted(set(SWEEP + [cutoff]))
    rows, freq_rows, sweep_rows = [], [], []
    for sid, g in df.groupby("set", sort=False):
        n, peaks, sig = count_reps(g, a.label)
        secs, dur = peaks / SAMPLING_HZ, len(g) / SAMPLING_HZ
        gaps = np.diff(secs)
        span = np.nanmax(sig) - np.nanmin(sig)
        level = lambda v: round(float((v - np.nanmin(sig)) / span), 2) if span else None  # noqa: E731
        who = g["participant"].iloc[0]
        rows.append(
            {
                "set": sid,
                "participant": who,
                "seconds": round(dur, 1),
                "true": true,
                "counted": n,
                "first_peak_s": round(float(secs[0]), 1) if n else None,
                "end_gap_s": round(float(dur - secs[-1]), 1) if n else None,
                "median_gap_s": round(float(np.median(gaps)), 1) if n > 1 else None,
                "max_gap_s": round(float(gaps.max()), 1) if n > 1 else None,
                "start_level": level(sig[0]),
                "end_level": level(sig[-1]),
            }
        )
        raw = add_magnitudes(g[SENSOR_COLS].astype(float).interpolate(limit_direction="both"))[column]
        freq_rows.append(
            {
                "set": sid,
                "participant": who,
                "reps_per_s": round(true / dur, 2),
                "dominant_hz": dominant_hz(raw),
                "cutoff_hz": cutoff,
            }
        )
        sweep_rows.append(
            {"set": sid, "participant": who, **{f"{c:g} Hz": count_reps(g, a.label, cutoff=c)[0] for c in cutoffs}}
        )

    out = pd.DataFrame(rows)
    print(f"signal: {column}, true reps per set: {true}\n")
    print(out.to_string(index=False))
    print(f"\nexact: {(out['true'] == out['counted']).sum()} of {len(out)} sets, "
          f"mean absolute error {(out['true'] - out['counted']).abs().mean():.2f}, "
          f"mean bias {(out['counted'] - out['true']).mean():+.2f}")

    print("\n== frequency check (is the rep frequency above the cutoff?) ==")
    print(pd.DataFrame(freq_rows).to_string(index=False))

    print(f"\n== cutoff sweep (current cutoff: {cutoff:g} Hz) ==")
    sweep = pd.DataFrame(sweep_rows)
    print(sweep.to_string(index=False))
    cols = [f"{c:g} Hz" for c in cutoffs]
    mae = (sweep[cols] - true).abs().mean().round(2)
    print("\nmean absolute error at each cutoff:")
    print(mae.to_string())


if __name__ == "__main__":
    main()
