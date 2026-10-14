"""Single source of truth for constants shared by training code and the app."""

SAMPLING_HZ = 5  # data is resampled to 200 ms, so Nyquist = 2.5 Hz

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]
SENSOR_COLS = ACC_COLS + GYR_COLS

# Feature settings, taken from build_features.py
LOWPASS_CUTOFF_HZ = 1.3
LOWPASS_ORDER = 5
PCA_COMPONENTS = 3  # computed, but the final feature set (feature_set4) does not use PCA
TEMPORAL_WS = 5  # 1 s
FREQ_WS = 14  # 2.8 s
N_CLUSTERS = 5
CLUSTER_COLS = ACC_COLS

EXERCISES = ["bench", "squat", "row", "ohp", "dead"]
REST = "rest"

# The app averages class probabilities over this many recent feature rows.
VOTE_ROWS = 10
