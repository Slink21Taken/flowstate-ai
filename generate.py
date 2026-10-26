"""
FlowState AI — Synthetic Training Data Generator
===================================================
Generates ~15,000 synthetic (task_features -> actual_duration) rows for
training the Stage 1+2 effort-prediction model.

Design notes (read this before changing distributions):
- Each synthetic "user" has FIXED LATENT TRAITS (velocity profile, peak
  window, stress sensitivity) that are NEVER exposed as training features.
  The model only sees population-level features (task_type, difficulty,
  deadline proximity, peak-window flag) — exactly what analyze_task()
  will see in production, since in real life you don't know a brand-new
  user's hidden traits upfront. That gap is intentional: it's what the
  per-user correction factor (Stage 4) exists to close later.
- Noise is injected at the per-task level (lognormal multiplicative noise)
  so the data is trainable but not perfectly deterministic — real outliers
  included (procrastination spikes, unusually fast runs, mis-estimated
  difficulty).
- Columns are named to map 1:1 onto the API contract fields.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # fixed seed = reproducible dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_USERS = 2300
N_ROWS = 100_000

TASK_TYPES = ["coding", "essay", "reading", "math", "creative"]

# Base duration (minutes) at difficulty=5, population average, per type.
# These are the "population prior" numbers — what Stage 2's model learns
# before any personalization is applied.
BASE_DURATION_AT_DIFF5 = {
    "coding": 75,
    "essay": 70,
    "reading": 45,
    "math": 60,
    "creative": 65,
}

# How much each extra difficulty point scales duration (multiplicative
# per point above/below 5). Harder tasks take disproportionately longer.
DIFFICULTY_SCALE_EXPONENT = {
    "coding": 1.22,
    "essay": 1.15,
    "reading": 1.08,
    "math": 1.25,
    "creative": 1.12,
}

# Difficulty distribution skew per type (Beta distribution shape params,
# scaled to 1-10). Coding/math skew harder; reading skews easier.
DIFFICULTY_BETA_PARAMS = {
    "coding": (5.0, 3.5),
    "essay": (4.0, 4.0),
    "reading": (2.5, 4.5),
    "math": (5.5, 3.2),
    "creative": (3.5, 4.0),
}

# Roughly how often each task type shows up in a student's workload.
TASK_TYPE_WEIGHTS = np.array([0.24, 0.20, 0.20, 0.22, 0.14])

PEAK_WINDOWS = ["morning", "evening"]

# ---------------------------------------------------------------------------
# Step 1: Generate synthetic users with fixed latent traits
# ---------------------------------------------------------------------------

def generate_users(n_users: int) -> pd.DataFrame:
    users = []
    for uid in range(n_users):
        # Velocity profile: lognormal per task type, centered at 1.0.
        # sigma=0.25 means most users are "normal" (0.8x-1.25x) but there's
        # a real tail of notably faster/slower students per subject.
        velocity = {
            f"velocity_{t}": float(RNG.lognormal(mean=0.0, sigma=0.25))
            for t in TASK_TYPES
        }

        peak_window = RNG.choice(PEAK_WINDOWS, p=[0.55, 0.45])

        # How much a high-difficulty task inflates actual time when the
        # user is working outside their peak window / under load.
        stress_sensitivity = float(np.clip(RNG.normal(1.0, 0.18), 0.5, 1.8))

        # Small fraction of users are chronic procrastinators / outlier
        # estimators — fat-tail multiplier applied rarely but meaningfully.
        is_outlier_prone = RNG.random() < 0.08

        users.append(
            {
                "user_id": f"u{uid}",
                "peak_window": peak_window,
                "stress_sensitivity": stress_sensitivity,
                "is_outlier_prone": is_outlier_prone,
                **velocity,
            }
        )
    return pd.DataFrame(users)


# ---------------------------------------------------------------------------
# Step 2: Generate tasks per user, compute actual_duration
# ---------------------------------------------------------------------------

def sample_difficulty(task_type: str) -> int:
    a, b = DIFFICULTY_BETA_PARAMS[task_type]
    raw = RNG.beta(a, b)  # 0-1
    scaled = 1 + raw * 9  # 1-10
    return int(np.clip(round(scaled), 1, 10))


def sample_deadline_hours() -> float:
    # Mixture: most tasks assigned 1-7 days out, some same-day crunch,
    # some assigned far in advance. Log-normal-ish with a floor.
    base = RNG.lognormal(mean=3.3, sigma=0.9)  # hours, roughly 5-300h mode
    hours = np.clip(base, 2, 336)  # 2h to 14 days
    return float(round(hours, 1))


def generate_rows(users: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    rows = []
    user_idx = RNG.integers(0, len(users), size=n_rows)

    for i in range(n_rows):
        user = users.iloc[user_idx[i]]
        task_type = RNG.choice(TASK_TYPES, p=TASK_TYPE_WEIGHTS)

        difficulty = sample_difficulty(task_type)
        deadline_hours_remaining = sample_deadline_hours()

        # Scheduled slot: randomly assign, then check against user's peak window
        scheduled_slot = RNG.choice(PEAK_WINDOWS, p=[0.5, 0.5])
        in_peak_window = scheduled_slot == user["peak_window"]

        # --- Population baseline duration for this (type, difficulty) ---
        base = BASE_DURATION_AT_DIFF5[task_type]
        exponent = DIFFICULTY_SCALE_EXPONENT[task_type]
        baseline_duration = base * (exponent ** (difficulty - 5))

        # --- Apply hidden user velocity (NOT a feature, only affects label) ---
        velocity_mult = user[f"velocity_{task_type}"]

        # --- Peak window effect: ~12% faster in peak window, ~8% slower out ---
        peak_mult = 0.88 if in_peak_window else 1.08

        # --- Deadline pressure effect: genuine crunch (<6h) adds rushed-but-
        #     sloppy time variance; very long runway (>120h) sometimes means
        #     task gets dragged out (Parkinson's law) ---
        if deadline_hours_remaining < 6:
            deadline_mult = float(RNG.normal(1.10, 0.12))  # rushed, some overhead
        elif deadline_hours_remaining > 120:
            deadline_mult = float(RNG.normal(1.06, 0.10))  # mild Parkinson's law
        else:
            deadline_mult = float(RNG.normal(1.0, 0.06))

        # --- Stress sensitivity kicks in harder on high-difficulty tasks ---
        if difficulty >= 8:
            stress_mult = user["stress_sensitivity"]
        else:
            stress_mult = 1.0 + (user["stress_sensitivity"] - 1.0) * 0.3

        # --- Base multiplicative noise (lognormal, realistic skew) ---
        noise_sigma = 0.16
        noise = RNG.lognormal(mean=0.0, sigma=noise_sigma)

        # --- Outlier injection: ~5% of ALL rows get a fat-tail event
        #     (interruption, distraction, sudden flow state, etc.), with
        #     outlier-prone users hitting this far more often ---
        outlier_roll = RNG.random()
        outlier_threshold = 0.18 if user["is_outlier_prone"] else 0.045
        is_outlier_row = outlier_roll < outlier_threshold
        if is_outlier_row:
            # Outliers skew mostly toward "took way longer", occasionally
            # "finished way faster than expected"
            if RNG.random() < 0.75:
                outlier_mult = float(RNG.uniform(1.6, 3.2))  # blew past estimate
            else:
                outlier_mult = float(RNG.uniform(0.35, 0.6))  # unusually fast
        else:
            outlier_mult = 1.0

        actual_duration = (
            baseline_duration
            * velocity_mult
            * peak_mult
            * deadline_mult
            * stress_mult
            * noise
            * outlier_mult
        )
        actual_duration = float(np.clip(actual_duration, 8, 1200))  # 8min-20h sanity bounds

        rows.append(
            {
                "task_id": f"t{i}",
                "user_id": user["user_id"],
                "task_type": task_type,
                "base_difficulty": difficulty,
                "deadline_hours_remaining": deadline_hours_remaining,
                "in_peak_window": in_peak_window,
                "predicted_duration_minutes_baseline": round(baseline_duration, 1),
                "actual_duration_minutes": round(actual_duration, 1),
                "is_outlier": is_outlier_row,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating {N_USERS} synthetic users...")
    users = generate_users(N_USERS)

    print(f"Generating {N_ROWS} synthetic task rows...")
    tasks = generate_rows(users, N_ROWS)

    # One-hot encode task_type for direct model consumption (keep the
    # original column too, for readability / scheduler logic downstream)
    onehot = pd.get_dummies(tasks["task_type"], prefix="type")
    model_ready = pd.concat([tasks, onehot], axis=1)

    # Save both: raw users (for reference/debugging) and the full dataset
    users.to_csv("synthetic_users.csv", index=False)
    tasks.to_csv("synthetic_tasks_raw.csv", index=False)
    model_ready.to_csv("synthetic_tasks_model_ready.csv", index=False)

    # --- Sanity report ---
    print("\n=== Sanity Report ===")
    print(f"Total rows: {len(tasks)}")
    print(f"Outlier rows: {tasks['is_outlier'].sum()} ({tasks['is_outlier'].mean()*100:.1f}%)")
    print("\nActual duration (minutes) by task_type:")
    print(tasks.groupby("task_type")["actual_duration_minutes"].describe()[["mean", "std", "min", "max"]].round(1))
    print("\nDifficulty distribution by task_type:")
    print(tasks.groupby("task_type")["base_difficulty"].describe()[["mean", "std"]].round(2))
    print("\nTask type counts:")
    print(tasks["task_type"].value_counts())
    print(f"\nPeak window match rate: {tasks['in_peak_window'].mean()*100:.1f}%")
    print("\nAny negative/zero durations?", (tasks["actual_duration_minutes"] <= 0).any())
    print("Any NaNs in dataset?", tasks.isna().any().any())

    print("\nSaved:")
    print("  synthetic_users.csv            (latent user traits, reference only — NOT for training)")
    print("  synthetic_tasks_raw.csv         (readable version)")
    print("  synthetic_tasks_model_ready.csv (one-hot encoded, ready for sklearn/XGBoost)")


if __name__ == "__main__":
    main()
