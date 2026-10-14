"""The first four tests compare the vectorised feature code with the original
loop-based classes from the Part 5 scripts (kept in tests/reference/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "reference"))
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis  # noqa: E402
from FrequencyAbstraction import FourierTransformation  # noqa: E402
from TemporalAbstraction import NumericalAbstraction  # noqa: E402

from src.config import FREQ_WS, SAMPLING_HZ, SENSOR_COLS, TEMPORAL_WS  # noqa: E402
from src.data import parse_filename  # noqa: E402
from src.features import PCAModel, frequency_features, low_pass, temporal_features  # noqa: E402
from src.model import MODEL_PATH, load_model, predict_rows, summarize  # noqa: E402
from src.rep_counter import benchmark, count_reps  # noqa: E402
from src.synthetic import make_training_df  # noqa: E402
from src.training import select_features, train_bundle  # noqa: E402

RNG = np.random.default_rng(1)


def test_low_pass_matches_original():
    x = RNG.normal(size=100)
    ref = LowPassFilter().low_pass_filter(pd.DataFrame({"x": x}), "x", SAMPLING_HZ, 1.3, order=5)
    np.testing.assert_allclose(low_pass(x, 1.3, order=5), ref["x_lowpass"].to_numpy())


def test_temporal_matches_original():
    df = pd.DataFrame(RNG.normal(size=(80, 3)), columns=["a", "b", "c"])
    ref, na = df.copy(), NumericalAbstraction()
    for c in df.columns:
        ref = na.abstract_numerical(ref, [c], TEMPORAL_WS, "mean")
        ref = na.abstract_numerical(ref, [c], TEMPORAL_WS, "std")
    pd.testing.assert_frame_equal(ref.drop(columns=df.columns), temporal_features(df, df.columns))


def test_frequency_matches_original():
    df = pd.DataFrame(RNG.normal(size=(60, 2)), columns=["a", "b"])
    ref = FourierTransformation().abstract_frequency(df.copy(), ["a", "b"], FREQ_WS, SAMPLING_HZ)
    pd.testing.assert_frame_equal(ref.drop(columns=df.columns), frequency_features(df, ["a", "b"]))


def test_pca_matches_original():
    df = pd.DataFrame(RNG.normal(size=(200, 6)) * [1, 2, 3, 40, 50, 60], columns=SENSOR_COLS)
    ref = PrincipalComponentAnalysis().apply_pca(df.copy(), SENSOR_COLS, 3)
    expected = ref[["pca_1", "pca_2", "pca_3"]].to_numpy()
    np.testing.assert_allclose(PCAModel(3).fit(df).transform(df), expected, atol=1e-9)


def test_parse_filename_strips_digits_not_letters():
    name = "A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    assert parse_filename(Path(name)) == ("A", "bench", "heavy")
    assert parse_filename(Path("B-squat-sit_MetaWear_x.csv"))[2] == "sit"


def test_low_pass_rejects_short_input():
    with pytest.raises(ValueError):
        low_pass(np.zeros(10))


def test_rep_counter_on_synthetic_sets():
    result, mae = benchmark(make_training_df(participants="AB", repeats=1))
    assert mae <= 1.0, result


def test_rep_counter_rest_is_zero():
    rest = make_training_df(participants="A", repeats=1).query("label == 'rest'")
    assert count_reps(rest, "rest")[0] == 0


@pytest.fixture(scope="module")
def trained():
    df = make_training_df(participants="ABC", repeats=1)
    return train_bundle(df, holdout="C", gridsearch=False), df[df["participant"] == "C"]


def test_feature_set_matches_notebook_feature_set4(trained):
    bundle, _ = trained
    cols = bundle["feature_columns"]
    # 6 basic + 2 squares + 16 temporal + 80 frequency (8 signals x (8 bins + weighted + pse)) + cluster
    assert len(cols) == 105 and len(set(cols)) == 105
    assert not any("pca" in c or "max_freq" in c or c == "duration" for c in cols)
    assert select_features(cols) == cols


def test_holdout_accuracy_on_synthetic(trained):
    bundle, _ = trained
    assert bundle["metrics"]["accuracy"] > 0.8


def test_predict_rows_warmup_and_probabilities(trained):
    bundle, held = trained
    one = held[held["set"] == held["set"].iloc[0]][SENSOR_COLS]
    proba = predict_rows(bundle, one)
    assert len(proba) == len(one)
    assert proba.iloc[:FREQ_WS].isna().all().all()  # not enough history yet
    assert proba.iloc[FREQ_WS:].notna().all().all()
    np.testing.assert_allclose(proba.iloc[FREQ_WS:].sum(axis=1), 1.0)
    assert summarize(proba, FREQ_WS)[0] is None
    assert summarize(proba, len(one))[0] == held["label"].iloc[0]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/make_demo_data.py first")
def test_app_runs_on_every_demo_set():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
    assert not at.exception
    demo = pd.read_csv(root / "data" / "sample" / "demo_sets.csv")
    assert len(at.sidebar.selectbox[0].options) == demo["set"].nunique()
    for set_id in demo["set"].unique():  # the dropdown's real values are set ids; labels are display-only
        at.sidebar.selectbox[0].select(int(set_id)).run()
        assert not at.exception, at.exception
    assert load_model()["labels"]