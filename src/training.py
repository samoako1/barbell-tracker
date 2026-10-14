"""Part 6: build a feature table and train the classifier bundle the app loads."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from .config import SENSOR_COLS
from .features import FeaturePipeline, iter_sets

META_COLS = ["label", "set", "participant"]
SQUARE_COLS = ["acc_r", "gyr_r"]

# Same grid as ClassificationAlgorithms.random_forest(gridsearch=True)
RF_GRID = [
    {
        "min_samples_leaf": [2, 10, 50, 100, 200],
        "n_estimators": [10, 50, 100],
        "criterion": ["gini", "entropy"],
    }
]


def select_features(columns):
    """feature_set4 from train_model.py: basic + squares + temporal + frequency + cluster.

    Uses the notebook's string filters. As in the notebook, PCA columns are not used,
    and the `*_max_freq` columns are excluded (the filter looks for "_freq_" with a
    trailing underscore, which "_max_freq" does not have).
    """
    columns = list(columns)
    time = [c for c in columns if "_temp_" in c]
    freq = [c for c in columns if "_freq_" in c or "_pse" in c]
    return SENSOR_COLS + SQUARE_COLS + time + freq + ["cluster"]


def feature_table(pipeline: FeaturePipeline, df: pd.DataFrame) -> pd.DataFrame:
    """One row per kept sample: features + label/set/participant.

    Mirrors the notebook: drop rows with NaN (warm-up windows), then keep every
    second row to reduce the overlap between neighbouring windows.
    """
    parts = []
    for sid, g in iter_sets(df):
        f = pipeline.transform(g)
        f["label"], f["set"], f["participant"] = g["label"].iloc[0], sid, g["participant"].iloc[0]
        parts.append(f)
    return pd.concat(parts).dropna().iloc[::2]


def train_bundle(df: pd.DataFrame, holdout=None, gridsearch=True, random_state=0) -> dict:
    """Fit the feature pipeline and the final random forest on every participant except `holdout`.

    gridsearch=True reproduces the notebook (5-fold grid search, accuracy). False trains
    a fixed forest (100 trees, min_samples_leaf=2) for quick runs and tests.
    """
    train_df = df if holdout is None else df[df["participant"] != holdout]
    pipeline = FeaturePipeline().fit(train_df)
    train = feature_table(pipeline, train_df)
    columns = select_features(train.columns)

    if gridsearch:
        search = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            RF_GRID,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ).fit(train[columns], train["label"])
        clf, best = search.best_estimator_, dict(search.best_params_)
    else:
        best = {"n_estimators": 100, "min_samples_leaf": 2, "criterion": "gini"}
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **best).fit(
            train[columns], train["label"]
        )

    metrics = {"best_params": best}
    if holdout is not None:
        test = feature_table(pipeline, df[df["participant"] == holdout])
        pred = clf.predict(test[columns])
        metrics.update(
            {
                "holdout_participant": holdout,
                "accuracy": float(accuracy_score(test["label"], pred)),
                "labels": list(clf.classes_),
                "confusion_matrix": confusion_matrix(test["label"], pred, labels=clf.classes_).tolist(),
                "n_test_rows": int(len(test)),
            }
        )

    return {
        "model": clf,
        "pipeline": pipeline,
        "feature_columns": columns,
        "labels": list(clf.classes_),
        "is_placeholder": False,
        "metrics": metrics,
    }
