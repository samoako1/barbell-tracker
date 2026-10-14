"""Part 5 feature engineering, ported from build_features.py, DataTransformation.py,
TemporalAbstraction.py and FrequencyAbstraction.py.

The temporal and frequency functions are vectorised rewrites of the loop-based
originals; tests/test_pipeline.py checks they return identical values.

Deliberate differences from the notebooks:
* Filtering is done per set. The notebook filtered the concatenated dataset, so
  the edges of each set were influenced by its neighbours.
* PCA statistics and KMeans are fitted once and stored, so inference reuses them.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import (
    ACC_COLS,
    CLUSTER_COLS,
    FREQ_WS,
    GYR_COLS,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_ORDER,
    N_CLUSTERS,
    PCA_COMPONENTS,
    SAMPLING_HZ,
    SENSOR_COLS,
    TEMPORAL_WS,
)


def min_samples(order: int) -> int:
    """filtfilt needs strictly more samples than its pad length, 3 * (order + 1)."""
    return 3 * (order + 1) + 1


def low_pass(values, cutoff_hz=LOWPASS_CUTOFF_HZ, order=LOWPASS_ORDER, fs=SAMPLING_HZ):
    """Zero-phase Butterworth low-pass (same as LowPassFilter.low_pass_filter)."""
    values = np.asarray(values, dtype=float)
    if len(values) < min_samples(order):
        raise ValueError(
            f"Need at least {min_samples(order)} samples to low-pass filter, got {len(values)}."
        )
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype="low", output="ba", analog=False)
    return filtfilt(b, a, values)


def add_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acc_r"] = np.sqrt((df[ACC_COLS] ** 2).sum(axis=1))
    df["gyr_r"] = np.sqrt((df[GYR_COLS] ** 2).sum(axis=1))
    return df


def temporal_features(df: pd.DataFrame, cols, ws=TEMPORAL_WS) -> pd.DataFrame:
    """Rolling mean and (population) std, as NumericalAbstraction.abstract_numerical."""
    out = {}
    for c in cols:
        roll = df[c].rolling(ws)
        out[f"{c}_temp_mean_ws_{ws}"] = roll.mean()
        out[f"{c}_temp_std_ws_{ws}"] = roll.std(ddof=0)  # original used np.std
    return pd.DataFrame(out, index=df.index)


def frequency_features(df: pd.DataFrame, cols, ws=FREQ_WS, fs=SAMPLING_HZ) -> pd.DataFrame:
    """Same values as FourierTransformation.abstract_frequency, without the Python loop.

    Row i uses samples i-ws .. i (ws + 1 samples); the first ws rows are NaN.
    """
    freqs = np.round(np.fft.rfftfreq(int(ws)) * fs, 3)
    n, nbins = len(df), len(freqs)
    out = {}
    for c in cols:
        max_freq = np.full(n, np.nan)
        weighted = np.full(n, np.nan)
        pse = np.full(n, np.nan)
        amps = np.full((n, nbins), np.nan)
        if n > ws:
            windows = sliding_window_view(df[c].to_numpy(dtype=float), ws + 1)
            real = np.fft.rfft(windows, axis=-1).real
            if real.shape[1] != nbins:
                raise ValueError("FREQ_WS must be even so the FFT and frequency bins line up.")
            with np.errstate(divide="ignore", invalid="ignore"):
                psd = real**2 / real.shape[1]
                pdf = psd / psd.sum(axis=1, keepdims=True)
                max_freq[ws:] = freqs[np.argmax(real, axis=1)]
                weighted[ws:] = (freqs * real).sum(axis=1) / real.sum(axis=1)
                pse[ws:] = -(np.log(pdf) * pdf).sum(axis=1)
            amps[ws:] = real
        out[f"{c}_max_freq"] = max_freq
        out[f"{c}_freq_weighted"] = weighted
        out[f"{c}_pse"] = pse
        for j, f in enumerate(freqs):
            out[f"{c}_freq_{f}_Hz_ws_{ws}"] = amps[:, j]
    return pd.DataFrame(out, index=df.index)


class PCAModel:
    """PrincipalComponentAnalysis.apply_pca, split into fit and transform.

    Inputs are normalised with (x - mean) / (max - min) using training statistics.
    """

    def __init__(self, n_components=PCA_COMPONENTS):
        self.n_components = n_components

    def fit(self, df: pd.DataFrame):
        self.cols_ = list(df.columns)
        self.mean_ = df.mean()
        self.range_ = df.max() - df.min()
        self.pca_ = PCA(n_components=self.n_components).fit(self._norm(df))
        self.pca_.components_[0] *= -1  # sign convention used in DataTransformation.py
        return self

    def _norm(self, df):
        return ((df[self.cols_] - self.mean_) / self.range_).to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pca_.transform(self._norm(df))


def iter_sets(df: pd.DataFrame, set_col="set"):
    """Yield (set_id, rows) for sets long enough to filter."""
    for sid, g in df.groupby(set_col, sort=False):
        if len(g) >= min_samples(LOWPASS_ORDER):
            yield sid, g


class FeaturePipeline:
    """raw sensor columns (one set) -> the feature table the classifier was trained on.

    Order matches build_features.py: interpolate, low-pass, PCA, magnitudes,
    temporal, frequency, cluster. Outlier removal is a training-time step and is
    not applied here.
    """

    @staticmethod
    def _smooth(df: pd.DataFrame) -> pd.DataFrame:
        x = df[SENSOR_COLS].astype(float).interpolate(limit_direction="both")
        return pd.DataFrame({c: low_pass(x[c].to_numpy()) for c in SENSOR_COLS}, index=df.index)

    def _features(self, smooth: pd.DataFrame) -> pd.DataFrame:
        out = smooth.copy()
        pcs = self.pca_.transform(smooth)
        for k in range(pcs.shape[1]):
            out[f"pca_{k + 1}"] = pcs[:, k]
        out = add_magnitudes(out)
        base = SENSOR_COLS + ["acc_r", "gyr_r"]
        return pd.concat([out, temporal_features(out, base), frequency_features(out, base)], axis=1)

    def fit(self, df: pd.DataFrame, set_col="set"):
        smooth = [self._smooth(g) for _, g in iter_sets(df, set_col)]
        self.pca_ = PCAModel().fit(pd.concat(smooth))
        feats = pd.concat([self._features(s) for s in smooth]).dropna().iloc[::2]
        self.kmeans_ = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=0).fit(
            feats[CLUSTER_COLS].to_numpy()
        )
        return self

    def transform(self, df_set: pd.DataFrame) -> pd.DataFrame:
        """Feature rows for one set, same length as the input (warm-up rows are NaN)."""
        feats = self._features(self._smooth(df_set))
        feats["cluster"] = self.kmeans_.predict(feats[CLUSTER_COLS].to_numpy())
        return feats
