"""Synthetic sensor data for the placeholder model, tests and offline demo.

NOT real exercise data. It follows the real schema (5 Hz, six sensor columns,
participant / label / category / set) so the same code path runs end to end.
"""
import numpy as np
import pandas as pd

from .config import EXERCISES, GYR_COLS, REST, SAMPLING_HZ, SENSOR_COLS

# label -> (motion direction, base rep frequency Hz, gyro axis, gyro amplitude deg/s)
PROFILES = {
    "bench": ((0.0, 0.0, 1.0), 0.33, "gyr_x", 20.0),
    "squat": ((0.0, 1.0, 0.0), 0.30, "gyr_z", 25.0),
    "row": ((1.0, 0.0, 0.0), 0.38, "gyr_x", 50.0),
    "ohp": ((0.0, 0.7, 0.7), 0.30, "gyr_y", 35.0),
    "dead": ((0.7, 0.7, 0.0), 0.28, "gyr_y", 20.0),
}


def _traits(participant: str) -> dict:
    r = np.random.default_rng(ord(participant[0]))
    return {"amp": r.uniform(0.85, 1.15), "freq": r.uniform(0.92, 1.08), "tilt": r.normal(0, 0.08, 3)}


def make_set(label, category, participant, set_id, rng) -> pd.DataFrame:
    traits = _traits(participant)
    if label == REST:
        n, wave, direction, gyr_axis, gyr_amp = 15 * SAMPLING_HZ, None, (0.0, 0.0, 1.0), None, 0.0
    else:
        direction, freq, gyr_axis, gyr_amp = PROFILES[label]
        reps = 5 if category == "heavy" else 10
        freq *= traits["freq"] * (1.0 if category == "heavy" else 1.15)
        n = int(round(reps / freq * SAMPLING_HZ))
        wave = np.sin(2 * np.pi * freq * np.arange(n) / SAMPLING_HZ)
    if wave is None:
        wave = np.zeros(n)

    u = np.array(direction) + traits["tilt"]
    u = u / np.linalg.norm(u)
    acc = u[None, :] * (1.0 + 0.5 * traits["amp"] * wave)[:, None] + rng.normal(0, 0.03, (n, 3))
    gyr = rng.normal(0, 3.0, (n, 3))
    if gyr_axis:
        gyr[:, GYR_COLS.index(gyr_axis)] += gyr_amp * traits["amp"] * wave

    start = pd.Timestamp("2019-01-11 10:00") + pd.Timedelta(hours=int(set_id))
    index = pd.date_range(start, periods=n, freq="200ms", name="epoch (ms)")
    df = pd.DataFrame(np.hstack([acc, gyr]), columns=SENSOR_COLS, index=index)
    df["label"], df["category"], df["participant"], df["set"] = label, category, participant, set_id
    return df


def make_training_df(participants="ABCD", repeats=2, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sets, set_id = [], 1
    for p in participants:
        for label in EXERCISES + [REST]:
            for category in ("sitting",) if label == REST else ("heavy", "medium"):
                for _ in range(repeats):
                    sets.append(make_set(label, category, p, set_id, rng))
                    set_id += 1
    return pd.concat(sets)
