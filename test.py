"""
FlowState AI — Chart Driver (v2)
=====================================
Corrected against the REAL train_model.py and feasibility-classifier
training script. Key fixes from v1:

  - #1 predicted-vs-actual: train_model.py never saved its test-set
    predictions anywhere. This script does NOT try to reload/re-predict
    -- instead it re-runs the exact same train/val/test split (same
    RANDOM_STATE, same split calls) against catboost_model.cbm, which
    gives byte-identical test rows without needing you to edit
    train_model.py at all.
  - #3/#3b feasibility: uses the REAL feature list from the training
    script (n_tasks, total_predicted_minutes, ..., n_completion_history),
    the REAL saved artifacts (feasibility_model.joblib,
    feasibility_scaler.joblib, feasibility_threshold.joblib -- a dict
    with key "infeasible_threshold", not a bare float), REAL StandardScaler
    transform before prediction, and works in P(infeasible) = 1 -
    P(on_time) space to match how the threshold was actually chosen.
  - #9/#11/#12 (archetype charts): DROPPED. synthetic_feasibility_data.csv
    (the file actually used for training) has no archetype column by
    design (the training script asserts against it). If you want these,
    you'd need synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv
    joined back to this one on (student_id, schedule_id) -- see the
    commented-out section at the bottom if that file exists for you.
  - #10 formula chart: replaced with an EMPIRICAL curve straight from
    synthetic_tasks_raw.csv, since train_model.py confirms there's no
    closed-form generator in this codebase to chart instead.

Run:
    python3 run_charts_v2.py
"""

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import charts as mc

BASE_URL = "http://localhost:5000"
RANDOM_STATE = 42  # must match train_model.py and the feasibility script exactly


def section(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")


# ---------------------------------------------------------------------------
# 1 & 2: CatBoost model charts
# ---------------------------------------------------------------------------
section("1 & 2: CatBoost model charts")
try:
    from catboost import CatBoostRegressor, Pool
    from pipeline import FEATURE_COLS  # the exact 8-feature list, same order as training

    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    # Feature importance straight from the loaded model
    # (needs a Pool at inference time for get_feature_importance in some
    # CatBoost versions -- rebuild train split below first, then call it)

    df = pd.read_csv("synthetic_tasks_raw.csv")

    def engineer_features(df):
        df = df.copy()
        df["log_deadline_hours"] = np.log1p(df["deadline_hours_remaining"])
        df["is_crunch"] = (df["deadline_hours_remaining"] < 6).astype(int)
        df["is_far_out"] = (df["deadline_hours_remaining"] > 120).astype(int)
        df["difficulty_squared"] = df["base_difficulty"] ** 2
        df["in_peak_window"] = df["in_peak_window"].astype(int)
        return df

    df = engineer_features(df)
    CATEGORICAL_COLS = ["task_type"]
    TARGET_COL = "actual_duration_minutes"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # EXACT same two-step split as train_model.py, same random_state ->
    # identical train/val/test rows without needing train_model.py edited.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    cat_idx = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    importances = model.get_feature_importance(train_pool)
    mc.chart_feature_importance(FEATURE_COLS, importances)

    preds = model.predict(test_pool)
    mc.chart_predicted_vs_actual(y_test.values, preds)

except Exception as e:
    print(f"[SKIP] model charts: {e}")


# ---------------------------------------------------------------------------
# 3 & 3b: Feasibility threshold tradeoff + confusion matrices
# ---------------------------------------------------------------------------
section("3 & 3b: Feasibility threshold charts")
try:
    FEASIBILITY_CSV = "synthetic_feasibility_data.csv"  # the real training file, not the DEBUG_ONLY one

    fm = joblib.load("feasibility_model.joblib")
    scaler = joblib.load("feasibility_scaler.joblib")
    threshold_data = joblib.load("feasibility_threshold.joblib")
    chosen_threshold = threshold_data["infeasible_threshold"]  # real key name

    df = pd.read_csv(FEASIBILITY_CSV)

    FEATURE_COLS = [
        "n_tasks", "total_predicted_minutes", "available_hours_to_earliest_deadline",
        "load_ratio", "concurrent_task_count", "peak_window_match_ratio",
        "historical_on_time_rate", "recent_slippage_trend", "n_completion_history",
    ]
    TARGET_COL = "completed_on_time"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    # Rebuild the SAME val split the training script used (stratified,
    # same random_state) so the threshold sweep below reflects the actual
    # validation set the real threshold was chosen on, not the full dataset.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    val_proba_on_time = fm.predict_proba(X_val_scaled)[:, 1]
    val_p_infeasible = 1 - val_proba_on_time
    val_y_infeasible = 1 - y_val.values

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred_infeasible = (val_p_infeasible >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            val_y_infeasible, pred_infeasible, average="binary",
            pos_label=1, zero_division=0,
        )
        precisions.append(p); recalls.append(r); f1s.append(f)

    mc.chart_feasibility_threshold_tradeoff(
        thresholds, precisions, recalls, f1s, chosen_threshold
    )

    # Confusion matrices on the TEST set (honest, untouched-until-now split),
    # default 0.5-on-P(infeasible) vs. the real chosen threshold.
    test_proba_on_time = fm.predict_proba(X_test_scaled)[:, 1]
    test_p_infeasible = 1 - test_proba_on_time
    test_y_infeasible = 1 - y_test.values

    pred_default = (test_p_infeasible >= 0.5).astype(int)
    pred_chosen = (test_p_infeasible >= chosen_threshold).astype(int)

    cm_default = confusion_matrix(test_y_infeasible, pred_default)
    cm_chosen = confusion_matrix(test_y_infeasible, pred_chosen)
    mc.chart_feasibility_confusion_matrix(cm_default, cm_chosen)

except FileNotFoundError as e:
    print(f"[SKIP] feasibility threshold charts: {e}")
except Exception as e:
    print(f"[SKIP] feasibility threshold charts: {e}")


# ---------------------------------------------------------------------------
# 4: Correction factor convergence (live API)
# ---------------------------------------------------------------------------
section("4: Correction factor convergence (live API)")
try:
    def complete(user_id, task_type, predicted, actual, task_id):
        return requests.post(f"{BASE_URL}/api/tasks/complete", json={
            "user_id": user_id, "task_id": task_id, "task_type": task_type,
            "predicted_duration_minutes": predicted, "actual_duration_minutes": actual,
        }, timeout=10)

    reliable_user = f"chart_reliable_{int(time.time())}"
    late_user = f"chart_late_{int(time.time())}"
    N = 10

    reliable_factors, late_factors = [], []
    for i in range(1, N + 1):
        r1 = complete(reliable_user, "essay", predicted=60, actual=60 + np.random.uniform(-3, 3),
                      task_id=f"r{i}")
        r2 = complete(late_user, "essay", predicted=60, actual=90 + np.random.uniform(-3, 3),
                      task_id=f"l{i}")
        reliable_factors.append(r1.json()["effective_factor"])
        late_factors.append(r2.json()["effective_factor"])

    mc.chart_correction_factor_convergence(range(1, N + 1), reliable_factors, late_factors)
except requests.exceptions.ConnectionError:
    print(f"[SKIP] correction factor convergence: could not reach {BASE_URL} — is Flask running?")
except Exception as e:
    print(f"[SKIP] correction factor convergence: {e}")


# ---------------------------------------------------------------------------
# 5: Feasibility before/after history (live API)
# ---------------------------------------------------------------------------
section("5: Feasibility before/after (live API)")
try:
    def generate(tasks, user_id, peak_window="morning"):
        return requests.post(f"{BASE_URL}/api/schedule/generate", json={
            "tasks": tasks, "user_id": user_id, "peak_window": peak_window,
        }, timeout=15)

    def complete(user_id, task_type, predicted, actual, task_id):
        return requests.post(f"{BASE_URL}/api/tasks/complete", json={
            "user_id": user_id, "task_id": task_id, "task_type": task_type,
            "predicted_duration_minutes": predicted, "actual_duration_minutes": actual,
        }, timeout=10)

    def make_task(task_id, hours_from_now=24, task_type="coding", difficulty=6):
        deadline = (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()
        return {"task_id": task_id, "title": f"Task {task_id}", "task_type": task_type,
                "deadline": deadline, "base_difficulty": difficulty}

    demo_user = f"chart_demo_{int(time.time())}"
    task = make_task("demo1", hours_from_now=24, task_type="coding")

    r_before = generate([task], user_id=demo_user)
    p_before = r_before.json()["feasibility"]["completion_probability"]

    for i in range(5):
        complete(demo_user, "coding", predicted=60, actual=60, task_id=f"hist{i}")

    r_after = generate([task], user_id=demo_user)
    p_after = r_after.json()["feasibility"]["completion_probability"]

    if p_before is not None and p_after is not None:
        mc.chart_feasibility_before_after(p_before, p_after)
    else:
        print(f"[SKIP] feasibility before/after: got None probability "
              f"(p_before={p_before}, p_after={p_after})")
except requests.exceptions.ConnectionError:
    print(f"[SKIP] feasibility before/after: could not reach {BASE_URL} — is Flask running?")
except Exception as e:
    print(f"[SKIP] feasibility before/after: {e}")


# ---------------------------------------------------------------------------
# 6: Pipeline diagram (no data needed)
# ---------------------------------------------------------------------------
section("6: Pipeline diagram")
mc.chart_pipeline_diagram()


# ---------------------------------------------------------------------------
# 7: Example schedule gantt (live API)
# ---------------------------------------------------------------------------
section("7: Schedule gantt (live API)")
try:
    def make_task(task_id, hours_from_now, task_type, difficulty):
        deadline = (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()
        return {"task_id": task_id, "title": f"Task {task_id}", "task_type": task_type,
                "deadline": deadline, "base_difficulty": difficulty}

    demo_tasks = [
        make_task("essay1", 30, "essay", 6),
        make_task("math1", 10, "math", 8),
        make_task("read1", 50, "reading", 3),
    ]
    r = requests.post(f"{BASE_URL}/api/schedule/generate", json={
        "tasks": demo_tasks, "user_id": f"chart_gantt_{int(time.time())}",
        "peak_window": "morning",
    }, timeout=15)
    r.raise_for_status()
    body = r.json()
    print("Schedule block keys present:", list(body["schedule"][0].keys()) if body["schedule"] else "EMPTY")
    mc.chart_schedule_gantt(body["schedule"])
except requests.exceptions.ConnectionError:
    print(f"[SKIP] schedule gantt: could not reach {BASE_URL} — is Flask running?")
except Exception as e:
    import traceback
    print(f"[SKIP] schedule gantt: {e}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 8: Performance scaling (live API, timed)
# ---------------------------------------------------------------------------
section("8: Performance scaling (live API)")
try:
    def make_task(task_id, hours_from_now):
        deadline = (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()
        return {"task_id": task_id, "title": f"Task {task_id}", "task_type": "essay",
                "deadline": deadline, "base_difficulty": 5}

    task_counts = [1, 10, 50, 100, 500, 1000]
    elapsed_times = []
    for n in task_counts:
        tasks = [make_task(f"t{i}", 24 + (i % 72)) for i in range(n)]
        start = time.time()
        r = requests.post(f"{BASE_URL}/api/schedule/generate", json={
            "tasks": tasks, "user_id": f"chart_perf_{n}_{int(time.time())}",
            "peak_window": "morning",
        }, timeout=60)
        elapsed = time.time() - start
        r.raise_for_status()
        elapsed_times.append(elapsed)
        print(f"  n={n}: {elapsed:.2f}s")

    mc.chart_performance_scaling(task_counts, elapsed_times)
except requests.exceptions.ConnectionError:
    print(f"[SKIP] performance scaling: could not reach {BASE_URL} — is Flask running?")
except Exception as e:
    print(f"[SKIP] performance scaling: {e}")


# ---------------------------------------------------------------------------
# 10: Empirical duration-by-difficulty curve (replaces the formula chart --
# train_model.py confirmed there's no closed-form generator in this repo)
# ---------------------------------------------------------------------------
section("10: Empirical duration by task type and difficulty")
try:
    import matplotlib.pyplot as plt

    df = pd.read_csv("synthetic_tasks_raw.csv")
    summary = (
        df.groupby(["task_type", "base_difficulty"])["predicted_duration_minutes_baseline"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tt in sorted(df["task_type"].unique()):
        sub = summary[summary["task_type"] == tt].sort_values("base_difficulty")
        ax.plot(sub["base_difficulty"], sub["predicted_duration_minutes_baseline"],
                marker="o", color=mc.TASK_TYPE_COLORS.get(tt, "#000"), label=tt)
    ax.set_xlabel("Base difficulty (1-10)")
    ax.set_ylabel("Mean baseline duration (min)")
    ax.set_title("Empirical Ground-Truth Duration by Task Type and Difficulty\n(from synthetic_tasks_raw.csv)")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs("charts", exist_ok=True)
    fig.savefig("charts/10_duration_by_difficulty.png", bbox_inches="tight")
    plt.close(fig)
    print("saved charts/10_duration_by_difficulty.png")
except Exception as e:
    print(f"[SKIP] duration by difficulty: {e}")


# ---------------------------------------------------------------------------
# 11 (optional, only if you have the DEBUG_ONLY file with archetype labels
# and it can be joined back to the real training file on
# (student_id, schedule_id)). Commented out by default since the public
# training file deliberately excludes this column.
# ---------------------------------------------------------------------------
# section("11: Archetype-labeled dataset composition (optional)")
# try:
#     debug_df = pd.read_csv("synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv")
#     real_df = pd.read_csv("synthetic_feasibility_data.csv")
#     merged = real_df.merge(
#         debug_df[["student_id", "schedule_id", "archetype"]],
#         on=["student_id", "schedule_id"], how="left",
#     )
#     mc.chart_dataset_composition(merged)
# except Exception as e:
#     print(f"[SKIP] archetype composition: {e}")


print(f"\n{'='*60}")
print("Done. Check ./charts/ for whatever generated successfully.")
print(f"{'='*60}")
