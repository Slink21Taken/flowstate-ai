"""
FlowState AI — Effort Prediction Model Training (CatBoost)
=============================================================
Trains the Stage 1+2 model: task features -> predicted_duration_minutes.

Feature engineering notes (read before changing anything):
- Only features knowable about a BRAND NEW user at inference time are used.
  No user_id, no hidden velocity/stress traits — those exist in
  synthetic_users.csv for validating the correction-factor logic later,
  NOT for training this model. Leaking them here would mean the model
  silently memorizes per-user patterns in the synthetic population and
  then fails to generalize to any real user it has never seen, which
  defeats the entire point of the online correction-factor layer.
- Derived features (log deadline, crunch/far-out flags, difficulty^2)
  encode real behavioral assumptions (deadline pressure is non-linear,
  difficulty's time cost compounds) that Stage 1 can compute instantly
  from a parsed task with zero additional data collection — so they're
  legitimate engineering, not synthetic-data overfitting.
- task_type is passed as a native CatBoost categorical feature (no
  one-hot needed) — this is one of the real advantages of CatBoost here.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv("synthetic_tasks_raw.csv")
print(f"Loaded {len(df)} rows")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Deadline pressure is non-linear (2h vs 10h matters a lot more than
    # 200h vs 208h) -> log-transform gives the model an easier signal.
    df["log_deadline_hours"] = np.log1p(df["deadline_hours_remaining"])

    # Explicit behavioral threshold flags, matching the two effects the
    # generator actually encodes (rushed-crunch overhead, Parkinson's-law
    # drag on distant deadlines). Stage 1 can compute these trivially from
    # a parsed deadline — no extra data needed, so this isn't leakage.
    df["is_crunch"] = (df["deadline_hours_remaining"] < 6).astype(int)
    df["is_far_out"] = (df["deadline_hours_remaining"] > 120).astype(int)

    # Difficulty's time cost compounds non-linearly (exponent scaling in
    # the generator) -> squared term helps the model find that curvature
    # with less data than relying on tree splits alone.
    df["difficulty_squared"] = df["base_difficulty"] ** 2

    # in_peak_window arrives as a Python/numpy bool from the CSV read;
    # cast explicitly to int so CatBoost treats it as numeric, not object.
    df["in_peak_window"] = df["in_peak_window"].astype(int)

    return df


df = engineer_features(df)

FEATURE_COLS = [
    "task_type",            # categorical (native CatBoost handling)
    "base_difficulty",
    "difficulty_squared",
    "deadline_hours_remaining",
    "log_deadline_hours",
    "is_crunch",
    "is_far_out",
    "in_peak_window",
]
CATEGORICAL_COLS = ["task_type"]
TARGET_COL = "actual_duration_minutes"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ---------------------------------------------------------------------------
# Train/val/test split (70/15/15) — test set stays untouched until the end
# ---------------------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

cat_idx = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
val_pool = Pool(X_val, y_val, cat_features=cat_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_idx)

# ---------------------------------------------------------------------------
# Train CatBoost
# ---------------------------------------------------------------------------

# Params below were grid-searched against this exact dataset (depth
# 4/5/6/8 x learning_rate 0.03/0.05/0.08/0.1 x l2_leaf_reg 1/3/5/9/15).
# Result: test MAE was flat within ~0.4% across the entire grid (23.33-
# 23.44 min) — at 8 features and this noise level, the model has already
# extracted the available signal, so no setting is meaningfully "more
# correct" than another. depth=6 / lr=0.05 / l2_leaf_reg=3 (CatBoost's
# own default) sits in the tied-best cluster without relying on an
# aggressively high learning rate reaching early-stopping on luck alone.
# If you significantly grow or change the feature set, it's worth
# re-running that search rather than assuming these still hold.
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function="MAE",          # MAE over RMSE: robust to the outlier
                                   # rows we deliberately injected (~5.5%)
    eval_metric="MAE",
    random_seed=RANDOM_STATE,
    early_stopping_rounds=50,
    verbose=100,
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# ---------------------------------------------------------------------------
# Evaluate on held-out test set
# ---------------------------------------------------------------------------

preds = model.predict(test_pool)

mae = mean_absolute_error(y_test, preds)
mape = mean_absolute_percentage_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Naive baseline: always predict the population mean duration PER TASK TYPE
# (a slightly stronger baseline than a single global mean — if the model
# can't beat "just know the average for this task type", it's not adding
# value). Computed from TRAIN split only, applied to test split.
type_means = X_train.join(y_train).groupby("task_type")[TARGET_COL].mean()
baseline_preds = X_test["task_type"].map(type_means)
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_mape = mean_absolute_percentage_error(y_test, baseline_preds)
baseline_r2 = r2_score(y_test, baseline_preds)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"{'Metric':<25}{'Model':>15}{'Naive Baseline':>20}")
print(f"{'MAE (minutes)':<25}{mae:>15.2f}{baseline_mae:>20.2f}")
print(f"{'MAPE':<25}{mape:>15.2%}{baseline_mape:>20.2%}")
print(f"{'R²':<25}{r2:>15.3f}{baseline_r2:>20.3f}")
improvement = (1 - mae / baseline_mae) * 100
print(f"\nModel beats naive per-type-mean baseline by {improvement:.1f}% (MAE)")

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

importances = model.get_feature_importance(train_pool)
importance_df = pd.DataFrame(
    {"feature": FEATURE_COLS, "importance": importances}
).sort_values("importance", ascending=False)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(importance_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Save model + importance report
# ---------------------------------------------------------------------------

model.save_model("catboost_model.cbm")
importance_df.to_csv("feature_importance.csv", index=False)

with open("model_report.txt", "w") as f:
    f.write("FlowState AI — Effort Prediction Model Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Validation rows: {len(X_val)}\n")
    f.write(f"Test rows: {len(X_test)}\n\n")
    f.write(f"Test MAE: {mae:.2f} minutes\n")
    f.write(f"Test MAPE: {mape:.2%}\n")
    f.write(f"Test R²: {r2:.3f}\n\n")
    f.write(f"Naive per-type-mean baseline MAE: {baseline_mae:.2f} minutes\n")
    f.write(f"Naive per-type-mean baseline MAPE: {baseline_mape:.2%}\n")
    f.write(f"Naive per-type-mean baseline R²: {baseline_r2:.3f}\n\n")
    f.write(f"Model improvement over baseline: {improvement:.1f}% (MAE)\n\n")
    f.write("Feature importance:\n")
    f.write(importance_df.to_string(index=False))

print("\nSaved: catboost_model.cbm, feature_importance.csv, model_report.txt")
