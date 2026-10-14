# Barbell Tracker

Classifies barbell exercises (bench press, squat, row, overhead press, deadlift, rest) and counts
reps from accelerometer and gyroscope data recorded with a MetaMotion wrist sensor.

**Live demo:** _add your Streamlit Cloud link here_

The demo replays real recorded sets from a participant the model never saw during training. Pick a
set, press Replay, and watch the predicted exercise, its confidence and the rep count update as the
playhead moves through the recording.

## Results

The model is evaluated by participant, not by row: the feature pipeline and classifier are fitted
without one person's data and scored on that person. Neighbouring rows within a set look almost
identical, so a random row split would leak and flatter the score.

**Leave-one-participant-out, all 5 participants (A to E):**

| Held out | Sets | Test rows | Accuracy | Rep MAE |
|---|---|---|---|---|
| A | 25 | 1291 | 99.6% | 0.60 |
| B | 9 | 359 | 88.3% | 1.44 |
| C | 17 | 622 | 94.1% | 0.29 |
| D | 9 | 463 | 100.0% | 0.78 |
| E | 25 | 1134 | 99.8% | 0.44 |

- Exercise accuracy: mean **96.4%** across participants (range 88.3% to 100%), 97.8% pooled over all rows.
- Rep counting: mean absolute error **0.71 reps** per set (range 0.29 to 1.44). True counts come from the
  set category in the dataset (5 for heavy sets, 10 for medium).
- Most common mistakes: bench press and overhead press confused with each other (42 and 10 rows),
  squat predicted as bench press (25 rows), and deadlift and row mixed up (7 and 2 rows).
- Participant B is the hardest case (88.3% accuracy, 1.44 reps off), and B has only 9 sets.

Reproduce with `python scripts/cross_validate.py` (fixed random forest per fold; add `--gridsearch`
to repeat the grid search in every fold, which is slow).

**The model in the app** is trained on participants B to E and evaluated on participant A (the
largest participant, and one of the easier ones): 99.6% row accuracy, rep MAE 0.6. The demo sets
come from A, so none of them were seen in training.

**Limits:**
- Five participants, two of them with only 9 sets, and one sensor placement.
- Feature importance shows the classifier leans mostly on the average orientation of the sensor over
  about 3 seconds (the 0 Hz FFT components and rolling means), so it recognises exercises largely by
  wrist posture. A different strap position could hurt accuracy.
- The squat and overhead-press rep cutoffs were tuned on all participants, so the rep error is slightly
  optimistic. Fast overhead-press sets are the weakest case for counting.
- Predicted probabilities are random-forest vote shares and are not calibrated.
- Data is resampled to 5 Hz, and outlier removal is not applied at inference.

## Pipeline

```
raw CSVs → read + tag by filename → merge acc/gyr → resample to 5 Hz            (src/data.py)
        → outlier removal (Chauvenet, training only)                              (notebook)
        → interpolate → low-pass (1.3 Hz) → PCA → magnitudes                      (src/features.py)
        → rolling mean/std (1 s) → FFT features (2.8 s) → KMeans cluster          (src/features.py)
        → random forest, grid-searched, on feature set 4 (no PCA)                 (src/training.py)
        → rep counter: exercise-specific low-pass + peak detection                (src/rep_counter.py)
```

The app loads one bundle (`models/model.joblib`) holding the fitted feature pipeline and the
classifier together, so it cannot compute different features than training did.

The classifier is a random forest grid-searched over `min_samples_leaf`, `n_estimators` and
`criterion` (5-fold CV, accuracy) on feature set 4: basic, squared, temporal, frequency and cluster
features (105 columns). PCA is computed but not used.

## Replay mode, and what it is not

Replay steps through a recorded set and reveals the prediction and rep count as the playhead
advances. The low-pass filters are zero-phase (`filtfilt`), so they use samples after the playhead,
exactly as in training. This is a faithful demo of the trained pipeline, not a real-time system.
Going live needs causal filters (or a short look-ahead buffer), retraining on those features, and
streaming windows.

## Run locally

Use Python 3.11.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pytest` (install it first: `pip install pytest`).

`models/model.joblib` only loads with the library versions pinned in `requirements.txt`.

## Retrain on your own data

```bash
python scripts/train.py --holdout A --demo-sets 2
```

This reads `data/interim/02_outliers_removed_chauvenets.pkl` (and `01_data_processed.pkl` for the
rep benchmark and demo sets), trains on every participant except `A`, evaluates on `A`, and writes
`models/model.joblib` plus `data/sample/demo_sets.csv`. Add `--no-gridsearch` for a quick run.

`scripts/tune_rep_cutoffs.py` and `scripts/check_reps.py` compare rep-counter settings.
Do not run `scripts/make_demo_data.py` on a real checkout: it writes a synthetic placeholder model
over `models/model.joblib`.

## Data

Sensor recordings come from the public MetaMotion dataset used in the "Tracking Barbell Exercises"
tutorial series. Raw data is not included in this repository. Only the demo sets in `data/sample/`
are.
