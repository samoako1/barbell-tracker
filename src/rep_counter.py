"""Part 7: repetition counting, ported from count_repetitions.py.

Low-pass a movement signal, then count local maxima. The signal and cutoff depend
on the exercise, so the app passes the predicted label.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .config import REST, SENSOR_COLS
from .features import add_magnitudes, low_pass, min_samples

REP_ORDER = 10
DEFAULT_PARAMS = ("acc_r", 0.4)
REP_PARAMS = {  # label -> (signal column, cutoff Hz)
    "row": ("gyr_x", 0.65),
    "squat": ("acc_r", 0.4),
    "ohp": ("acc_r", 0.45),
}


def rep_params(label):
    return REP_PARAMS.get(label, DEFAULT_PARAMS)


def count_reps(df: pd.DataFrame, label: str, cutoff=None):
    """Return (n_reps, peak_positions, filtered_signal) for one set.

    Peak positions are integer row positions. Rest, or a set too short to filter,
    returns zero reps and an all-NaN signal. `cutoff` overrides the label's default
    low-pass cutoff in Hz (used by scripts/check_reps.py to test other values).
    """
    n = len(df)
    if label == REST or n < min_samples(REP_ORDER):
        return 0, np.array([], dtype=int), np.full(n, np.nan)
    column, default_cutoff = rep_params(label)
    cutoff = default_cutoff if cutoff is None else cutoff
    data = add_magnitudes(df[SENSOR_COLS].astype(float).interpolate(limit_direction="both"))
    signal = low_pass(data[column].to_numpy(), cutoff, order=REP_ORDER)
    peaks = argrelextrema(signal, np.greater)[0]
    return len(peaks), peaks, signal


def true_reps(category: str) -> int:
    """The dataset encodes reps in the set category: heavy = 5, otherwise 10."""
    return 5 if category == "heavy" else 10


def benchmark(df: pd.DataFrame):
    """Predicted vs true reps per set (using the true label), plus mean absolute error."""
    rows = []
    for sid, g in df[df["label"] != REST].groupby("set", sort=False):
        label, category = g["label"].iloc[0], g["category"].iloc[0]
        pred = count_reps(g, label)[0]
        rows.append(
            {"set": sid, "label": label, "category": category, "reps": true_reps(category), "reps_pred": pred}
        )
    result = pd.DataFrame(rows)
    mae = float((result["reps"] - result["reps_pred"]).abs().mean()) if len(result) else float("nan")
    return result, mae
