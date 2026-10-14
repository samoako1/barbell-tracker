"""Barbell exercise classifier + rep counter

    streamlit run app.py
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import ACC_COLS, GYR_COLS, REST, SAMPLING_HZ, SENSOR_COLS
from src.model import load_model, predict_rows, summarize
from src.rep_counter import count_reps, rep_params, true_reps

ROOT = Path(__file__).resolve().parent
DEMO_CSV = ROOT / "data" / "sample" / "demo_sets.csv"

LABEL_NAMES = {
    "bench": "Bench press",
    "squat": "Squat",
    "row": "Barbell row",
    "ohp": "Overhead press",
    "dead": "Deadlift",
    "rest": "Rest",
}


def pretty(label):
    return LABEL_NAMES.get(label, label)


st.set_page_config(page_title="Barbell Tracker", layout="wide")
st.title("Barbell Exercise Classifier and Rep Counter")
st.caption(
    f"MetaMotion accelerometer + gyroscope at {SAMPLING_HZ} Hz → low-pass → PCA → temporal and "
    "frequency features → random forest → rep counter. Replay reveals results as the playhead "
    "advances, but the filters are zero-phase (non-causal) over the whole recording, exactly as in "
    "training. A real-time version needs causal filters and retraining."
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_demo():
    return pd.read_csv(DEMO_CSV, index_col="timestamp", parse_dates=True)


@st.cache_data
def classify(_bundle, sensors):
    return predict_rows(_bundle, sensors)


try:
    bundle = get_model()
except FileNotFoundError:
    st.error("No model found. Run `python scripts/make_demo_data.py` or `python scripts/train.py`.")
    st.stop()

if bundle["is_placeholder"]:
    st.warning("Placeholder model trained on synthetic data. Run scripts/train.py before presenting.")

# ---- data source -----------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Demo sets", "Upload CSV"])
    truth, truth_reps = None, None
    if source == "Demo sets":
        demo = get_demo()
        info = demo.groupby("set")[["label", "category"]].first()
        names = {s: f"{pretty(r['label'])}, {r['category']}" for s, r in info.iterrows()}
        totals = pd.Series(names).value_counts()
        seen = {}
        for s, name in list(names.items()):  # number repeats: "Squat, heavy (1)", "(2)"
            if totals[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                names[s] = f"{name} ({seen[name]})"
        set_id = st.selectbox(
            "Recorded set", sorted(info.index, key=lambda s: names[s]), format_func=names.get
        )
        chosen = demo[demo["set"] == set_id]
        truth = info.loc[set_id, "label"]
        truth_reps = None if truth == REST else true_reps(info.loc[set_id, "category"])
        sensors = chosen[SENSOR_COLS].reset_index(drop=True)
    else:
        set_id = "upload"
        up = st.file_uploader(f"CSV at {SAMPLING_HZ} Hz with columns {', '.join(SENSOR_COLS)}", type="csv")
        if up is None:
            st.info("Upload a CSV, or switch to Demo sets.")
            st.stop()
        raw = pd.read_csv(up)
        missing = [c for c in SENSOR_COLS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        sensors = raw[SENSOR_COLS].reset_index(drop=True)
    speed = st.slider("Replay speed", 1, 10, 4)

    with st.expander("Model evaluation"):
        m = bundle.get("metrics") or {}
        if m.get("accuracy") is not None:
            st.write(
                f"Held-out participant {m['holdout_participant']}: "
                f"**{m['accuracy']:.1%}** row accuracy"
                + (f", rep count MAE {m['rep_mae']}" if m.get("rep_mae") is not None else "")
            )
            st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]))
        else:
            st.write("No evaluation stored with this model.")

n = len(sensors)
if n < 2:
    st.error("Need at least two samples.")
    st.stop()
t = np.arange(n) / SAMPLING_HZ

try:
    proba = classify(bundle, sensors)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---- replay -------------------------------------------------------------------
col_a, col_b = st.columns([1, 4])
play = col_a.button("▶ Replay")
head = col_b.slider("Playhead (samples)", 1, n, n, key=f"head_{source}_{set_id}")

metrics = st.empty()
chart = st.empty()
rep_cache = {}
frames = itertools.count()  # unique key per chart draw; the last replay frame is drawn twice


def reps_for(label):
    if label not in rep_cache:
        rep_cache[label] = count_reps(sensors, label)
    return rep_cache[label]


def render(upto: int):
    label, probs = summarize(proba, upto)
    shown_peaks = np.array([], dtype=int)
    signal = None
    if label is not None and label != REST:
        _, peaks, signal = reps_for(label)
        shown_peaks = peaks[peaks < upto]

    with metrics.container():
        m1, m2, m3 = st.columns(3)
        if label is None:
            m1.metric("Predicted exercise", "warming up…")
        else:
            m1.metric(
                "Predicted exercise", pretty(label), f"{probs[label]:.0%} confidence",
                delta_color="off", delta_arrow="off",
            )
        m2.metric("Reps counted", len(shown_peaks))
        if truth is not None:
            m3.metric(
                "Ground truth", pretty(truth), None if truth_reps is None else f"{truth_reps} reps",
                delta_color="off", delta_arrow="off",
            )
        else:
            m3.metric("Ground truth", "n/a")

    if signal is not None:
        column, cutoff = rep_params(label)
        rep_title = f"Rep signal: {column}, {cutoff} Hz low-pass"
    else:
        rep_title = "Rep signal"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        row_heights=[0.55, 0.45], subplot_titles=("Accelerometer (raw)", rep_title),
    )
    for c in ACC_COLS:
        fig.add_trace(go.Scatter(x=t[:upto], y=sensors[c].iloc[:upto], name=c, line=dict(width=1)), row=1, col=1)
    if signal is not None:
        fig.add_trace(go.Scatter(x=t[:upto], y=signal[:upto], name="filtered", line=dict(width=2)), row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=t[shown_peaks], y=signal[shown_peaks], mode="markers", name="rep",
                marker=dict(size=11, symbol="x", color="red"),
            ),
            row=2, col=1,
        )
    fig.update_xaxes(range=[0, t[-1]], title_text="seconds", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
    chart.plotly_chart(fig, width="stretch", key=f"chart_{next(frames)}")


if play:
    for upto in range(1, n + 1, speed):
        render(upto)
        time.sleep(0.12)
    render(n)
else:
    render(head)

with st.expander("Gyroscope signals"):
    st.line_chart(sensors[GYR_COLS])
