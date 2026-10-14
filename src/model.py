"""Load the saved bundle and classify a recording.

The bundle (see training.train_bundle) holds the fitted feature pipeline and the
classifier together, so the app can never compute different features than training.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import VOTE_ROWS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
REQUIRED_KEYS = ("model", "pipeline", "feature_columns", "labels", "is_placeholder")


def load_model(path: Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise KeyError(f"model bundle missing {missing}")
    return bundle


def predict_rows(bundle: dict, sensors: pd.DataFrame) -> pd.DataFrame:
    """Class probabilities per row (one column per label). Warm-up rows are NaN."""
    feats = bundle["pipeline"].transform(sensors)
    X = feats[bundle["feature_columns"]]
    ok = X.notna().all(axis=1).to_numpy()
    out = np.full((len(X), len(bundle["labels"])), np.nan)
    if ok.any():
        out[ok] = bundle["model"].predict_proba(X[ok])
    return pd.DataFrame(out, columns=bundle["labels"])


def summarize(proba: pd.DataFrame, upto: int, vote_rows: int = VOTE_ROWS):
    """Average probabilities over the last `vote_rows` valid rows before `upto`."""
    recent = proba.iloc[:upto].dropna().tail(vote_rows)
    if recent.empty:
        return None, None
    mean = recent.mean()
    return mean.idxmax(), mean.to_dict()
