"""Part 2: read raw MetaMotion CSVs, tag them from the filename, merge, resample."""
from pathlib import Path

import pandas as pd

from .config import SENSOR_COLS

DATA_PATH = Path("data/raw/MetaMotion")
DROP_COLS = ["epoch (ms)", "elapsed (s)"]

SAMPLING = {
    **{c: "mean" for c in SENSOR_COLS},
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}


def parse_filename(path: Path):
    """e.g. A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_..._Accelerometer_12.500Hz_1.4.4.csv"""
    stem = path.name.split("_MetaWear_")[0]  # "A-bench-heavy2-rpe8"
    participant, label, category = stem.split("-")[:3]
    return participant, label, category.rstrip("0123456789")


def read_data_from_files(data_path: Path = DATA_PATH):
    frames = {"Accelerometer": [], "Gyroscope": []}
    counters = {"Accelerometer": 1, "Gyroscope": 1}

    for f in sorted(Path(data_path).glob("*.csv")):
        sensor = "Accelerometer" if "Accelerometer" in f.name else "Gyroscope"
        participant, label, category = parse_filename(f)

        df = pd.read_csv(f)
        df.index = pd.to_datetime(df["epoch (ms)"], unit="ms")
        # "time (01:00)" encodes the exporter's UTC offset, so match by prefix
        df = df.drop(columns=DROP_COLS + [c for c in df.columns if c.startswith("time (")])
        df = df.assign(
            participant=participant, label=label, category=category, set=counters[sensor]
        )
        counters[sensor] += 1
        frames[sensor].append(df)

    return pd.concat(frames["Accelerometer"]), pd.concat(frames["Gyroscope"])


def merge_and_resample(acc_df: pd.DataFrame, gyr_df: pd.DataFrame, rule: str = "200ms"):
    data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
    data.columns = SENSOR_COLS + ["participant", "label", "category", "set"]

    # Resample per day so gaps between sessions don't create giant empty ranges
    days = [g for _, g in data.groupby(pd.Grouper(freq="D"))]
    out = pd.concat([d.resample(rule).apply(SAMPLING).dropna() for d in days])
    out["set"] = out["set"].astype("int")
    return out
