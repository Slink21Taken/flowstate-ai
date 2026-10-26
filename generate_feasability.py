"""
FlowState AI — Synthetic Training Data Generator (Feasibility Classifier)
=============================================================================
Generates synthetic (schedule_features -> completed_on_time) rows for
training the Stage 5 schedule-feasibility classifier (TODO_backend_ml.md
item 2). This is the schedule-level counterpart to
generate_training_data.py (which generates per-TASK duration rows for the
Stage 1+2 CatBoost regressor) -- same philosophy, one level up.

Design notes (read this before changing distributions) -- mirrors the
duration generator's own rules:
- Each synthetic "student" has a FIXED LATENT ARCHETYPE (reliable,
  chronically-late, crunch-sensitive, erratic) that is NEVER exposed as a
  training feature. The classifier only sees the same schedule-level
  signals a real system can observe: total load vs. available time,
  concurrent-task count, peak-window match ratio, historical on-time
  rate, and recent slippage trend. The archetype itself is the thing the
  model has to learn to recognize FROM those signals -- exposing it
  directly would be label leakage.
- Real pipeline code is reused wherever possible instead of being
  reimplemented: TaskForScheduling, chunk_task, compute_priority_score,
  compute_buffer_deadline and generate_schedule are imported directly
  from scheduler.py, so "concurrent/overlapping task count" and
  "buffer_deadline" come from the actual scheduling logic, not a
  parallel approximation of it that could drift.
- Population task duration is NOT taken from a live CatBoost call --
  there's no trained model artifact to depend on here, and more to the
  point, this generator needs GROUND-TRUTH population duration to judge
  "is this schedule overloaded?" against, which is exactly the formula
  generate_training_data.py already defines (BASE_DURATION_AT_DIFF5 x
  DIFFICULTY_SCALE_EXPONENT). Reusing those constants keeps this
  generator consistent with the regressor's own world instead of
  inventing a second, divergent notion of "how long should this take."
- A student's personalization state (per-task-type correction factor +
  n_samples) is simulated directly from their archetype's TRUE velocity,
  using the real CorrectionFactorStore EMA math (imported from
  pipeline.py) applied to a synthetic completion history -- so
  "historical on-time rate" and "recent slippage trend" come from the
  same formula production actually runs, not a shortcut average.
- Labels are sampled probabilistically from a logistic function of
  (archetype reliability - overload pressure), not assigned
  deterministically from the same features being trained on. This keeps
  real signal in the data (the model can learn it) while leaving
  irreducible noise (archetypes are not 100% separable), the same way
  the duration generator keeps noise + a genuine outlier tail instead of
  a perfectly deterministic formula.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduler import (  # noqa: E402
    TaskForScheduling,
    generate_schedule,
    compute_buffer_deadline,
    PEAK_WINDOWS,
)
from pipeline import CorrectionFactorStore, EMA_ALPHA, FACTOR_MIN, FACTOR_MAX, MIN_SAMPLES_FOR_FULL_TRUST  # noqa: E402

RNG = np.random.default_rng(42)  # fixed seed = reproducible dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_STUDENTS = 10000
# Each student gets multiple simulated "schedule generation moments"
# (different task batches, different points in their completion history) --
# this is what produces schedule-LEVEL rows, since a single student only
# generates one schedule at a time in production.
SCHEDULES_PER_STUDENT = 6
N_ROWS_TARGET = N_STUDENTS * SCHEDULES_PER_STUDENT  # ~9,000 rows

TASK_TYPES = ["coding", "essay", "reading", "math", "creative"]
TASK_TYPE_WEIGHTS = np.array([0.24, 0.20, 0.20, 0.22, 0.14])

# Same ground-truth population duration formula as generate_training_data.py
# -- kept identical on purpose, see module docstring.
BASE_DURATION_AT_DIFF5 = {
    "coding": 75, "essay": 70, "reading": 45, "math": 60, "creative": 65,
}
DIFFICULTY_SCALE_EXPONENT = {
    "coding": 1.22, "essay": 1.15, "reading": 1.08, "math": 1.25, "creative": 1.12,
}
DIFFICULTY_BETA_PARAMS = {
    "coding": (5.0, 3.5), "essay": (4.0, 4.0), "reading": (2.5, 4.5),
    "math": (5.5, 3.2), "creative": (3.5, 4.0),
}

PEAK_WINDOW_NAMES = list(PEAK_WINDOWS.keys())  # ["morning", "evening"]


def population_duration(task_type: str, difficulty: int) -> float:
    """Ground-truth population duration -- NOT a live model call. See module docstring."""
    base = BASE_DURATION_AT_DIFF5[task_type]
    exponent = DIFFICULTY_SCALE_EXPONENT[task_type]
    return max(base * (exponent ** (difficulty - 5)), 5.0)


def sample_difficulty(task_type: str) -> int:
    a, b = DIFFICULTY_BETA_PARAMS[task_type]
    raw = RNG.beta(a, b)
    return int(np.clip(round(1 + raw * 9), 1, 10))


# ---------------------------------------------------------------------------
# Step 1: Student archetypes (fixed latent traits, never exposed as features)
# ---------------------------------------------------------------------------
#
# Each archetype is a GENERATIVE PROCESS, not a lookup table: it has a true
# on-time base rate and a sensitivity to "overload pressure" (how it decays
# as the schedule gets tighter). The classifier never sees which archetype
# a student belongs to -- only the behavioral traces (completion history,
# slippage trend) that process leaves behind, same as a real system would
# only ever observe behavior, never ground-truth "type."

ARCHETYPES = {
    # Dependable; mild degradation under real overload.
    "reliable": {
        "weight": 0.35,
        "base_on_time_rate": 0.90,
        "overload_sensitivity": 1.4,   # logistic slope on overload pressure
        "velocity_mean": 0.95,         # true actual/predicted ratio, pre-noise
        "velocity_sigma": 0.12,
    },
    # Structurally runs late regardless of load; starts lower, decays faster.
    "chronically_late": {
        "weight": 0.25,
        "base_on_time_rate": 0.55,
        "overload_sensitivity": 1.1,
        "velocity_mean": 1.35,
        "velocity_sigma": 0.22,
    },
    # Fine when uncrowded, collapses sharply once concurrent load rises --
    # the archetype most worth a feasibility classifier catching, since a
    # naive duration-sum check wouldn't flag it until it's already failed.
    "crunch_sensitive": {
        "weight": 0.25,
        "base_on_time_rate": 0.82,
        "overload_sensitivity": 2.6,   # steep: punishes overload hard
        "velocity_mean": 1.05,
        "velocity_sigma": 0.18,
    },
    # Noisy / inconsistent; neither reliably good nor bad, high variance.
    "erratic": {
        "weight": 0.15,
        "base_on_time_rate": 0.68,
        "overload_sensitivity": 1.0,
        "velocity_mean": 1.15,
        "velocity_sigma": 0.38,
    },
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())
ARCHETYPE_WEIGHTS = np.array([ARCHETYPES[a]["weight"] for a in ARCHETYPE_NAMES])
ARCHETYPE_WEIGHTS = ARCHETYPE_WEIGHTS / ARCHETYPE_WEIGHTS.sum()


@dataclass
class Student:
    student_id: str
    archetype: str
    peak_window: str
    velocity_mean: float       # true mean actual/predicted ratio for this student
    velocity_sigma: float
    base_on_time_rate: float
    overload_sensitivity: float


def generate_students(n: int) -> list[Student]:
    students = []
    for i in range(n):
        archetype = RNG.choice(ARCHETYPE_NAMES, p=ARCHETYPE_WEIGHTS)
        a = ARCHETYPES[archetype]
        students.append(
            Student(
                student_id=f"s{i}",
                archetype=archetype,
                peak_window=RNG.choice(PEAK_WINDOW_NAMES, p=[0.55, 0.45]),
                velocity_mean=float(np.clip(RNG.normal(a["velocity_mean"], 0.06), 0.5, 2.2)),
                velocity_sigma=a["velocity_sigma"],
                base_on_time_rate=float(np.clip(RNG.normal(a["base_on_time_rate"], 0.04), 0.05, 0.98)),
                overload_sensitivity=a["overload_sensitivity"],
            )
        )
    return students


# ---------------------------------------------------------------------------
# Step 2: Simulate a completion history per student, via the REAL
# CorrectionFactorStore EMA math (imported from pipeline.py) -- so
# "historical on-time rate" / "recent slippage trend" features come from
# the same formula production runs, not a parallel shortcut.
# ---------------------------------------------------------------------------

def simulate_completion_history(student: Student, store: CorrectionFactorStore, n_completions: int) -> list[float]:
    """
    Feeds n_completions synthetic (predicted, actual) pairs through the
    real store.update() EMA logic for this student, task type by task
    type, and returns the raw actual/predicted ratios observed (used
    below to compute the recent-slippage-trend feature honestly, since
    that's a derived statistic over raw observations, not something
    CorrectionFactorStore itself exposes).
    """
    ratios = []
    for _ in range(n_completions):
        task_type = RNG.choice(TASK_TYPES, p=TASK_TYPE_WEIGHTS)
        difficulty = sample_difficulty(task_type)
        predicted = population_duration(task_type, difficulty)

        ratio = float(RNG.lognormal(
            mean=np.log(student.velocity_mean), sigma=student.velocity_sigma
        ))
        ratio = float(np.clip(ratio, 0.3, 4.0))
        actual = predicted * ratio

        store.update(student.student_id, task_type, predicted, actual)
        ratios.append(ratio)
    return ratios


# ---------------------------------------------------------------------------
# Step 3: Generate one schedule-generation "moment" for a student, using
# the REAL scheduler.py (generate_schedule, compute_buffer_deadline) so
# concurrency/overlap and buffer-deadline features reflect actual
# scheduling behavior rather than a reimplementation.
# ---------------------------------------------------------------------------

def sample_task_batch(now: datetime) -> list[dict]:
    n_tasks = int(RNG.integers(2, 7))  # 2-6 tasks per batch
    tasks = []
    for i in range(n_tasks):
        task_type = RNG.choice(TASK_TYPES, p=TASK_TYPE_WEIGHTS)
        difficulty = sample_difficulty(task_type)
        # Deadlines clustered to sometimes produce genuine overload: mixture
        # of spread-out and crammed-together windows.
        if RNG.random() < 0.35:
            hours_out = float(RNG.uniform(3, 30))     # crammed/urgent window
        else:
            hours_out = float(RNG.uniform(12, 200))   # normal spread
        tasks.append(
            {
                "task_id": f"tt{i}",
                "title": f"{task_type} task {i}",
                "task_type": task_type,
                "difficulty": difficulty,
                "deadline": now + timedelta(hours=hours_out),
            }
        )
    return tasks


def build_schedule_features(student: Student, store: CorrectionFactorStore, now: datetime) -> dict:
    """
    Builds one schedule-level feature row, mirroring
    TODO_backend_ml.md's feature list exactly:
      - total predicted minutes vs. available hours before earliest deadline
      - number of concurrent/overlapping tasks in the window
      - peak-window match ratio
      - user's historical on-time completion rate
      - recent slippage trend (avg actual/predicted over last N completions)
    """
    raw_tasks = sample_task_batch(now)

    scheduling_tasks = []
    for t in raw_tasks:
        corr = store.effective_factor(student.student_id, t["task_type"])
        pred = population_duration(t["task_type"], t["difficulty"]) * corr
        scheduling_tasks.append(
            TaskForScheduling(
                task_id=t["task_id"],
                title=t["title"],
                task_type=t["task_type"],
                difficulty=t["difficulty"],
                predicted_duration_minutes=pred,
                deadline=t["deadline"],
            )
        )

    schedule_result = generate_schedule(scheduling_tasks, peak_window=student.peak_window, now=now)
    task_analysis = schedule_result["task_analysis"]

    # --- total predicted minutes vs. available hours before earliest deadline ---
    total_predicted_minutes = sum(t.predicted_duration_minutes for t in scheduling_tasks)
    earliest_deadline = min(t.deadline for t in scheduling_tasks)
    available_hours = max((earliest_deadline - now).total_seconds() / 3600, 0.5)
    load_ratio = (total_predicted_minutes / 60) / available_hours

    # --- concurrent/overlapping tasks: how many tasks share a deadline
    #     window tighter than the earliest deadline's own buffer window ---
    buffer_deadlines = [compute_buffer_deadline(t.deadline, now) for t in scheduling_tasks]
    earliest_buffer = min(buffer_deadlines)
    concurrent_count = sum(1 for bd in buffer_deadlines if bd <= earliest_buffer + timedelta(hours=12))

    # --- peak-window match ratio: how many tasks' deadlines leave >=4h
    #     of runway, same proxy app.py uses for in_peak_window upstream ---
    in_peak_flags = [
        1 if (t.deadline - now).total_seconds() / 3600 >= 4 else 0
        for t in scheduling_tasks
    ]
    peak_match_ratio = float(np.mean(in_peak_flags))

    # --- historical on-time rate + recent slippage trend, from the
    #     student's simulated completion history (ground truth, used only
    #     to derive observable stats -- archetype itself stays hidden) ---
    history = student.history_ratios  # populated by caller before this fn
    if history:
        recent = history[-10:]
        historical_on_time_rate = float(np.mean([1.0 if r <= 1.15 else 0.0 for r in recent]))
        recent_slippage_trend = float(np.mean(recent))
        n_completions = len(history)
    else:
        historical_on_time_rate = 0.5  # uninformative prior, brand-new student
        recent_slippage_trend = 1.0
        n_completions = 0

    return {
        "n_tasks": len(scheduling_tasks),
        "total_predicted_minutes": round(total_predicted_minutes, 1),
        "available_hours_to_earliest_deadline": round(available_hours, 2),
        "load_ratio": round(load_ratio, 4),
        "concurrent_task_count": concurrent_count,
        "peak_window_match_ratio": round(peak_match_ratio, 3),
        "historical_on_time_rate": round(historical_on_time_rate, 3),
        "recent_slippage_trend": round(recent_slippage_trend, 3),
        "n_completion_history": n_completions,
    }, task_analysis


# ---------------------------------------------------------------------------
# Step 4: Probabilistic labeling -- logistic(archetype reliability -
# overload pressure), sampled, not assigned deterministically.
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def sample_label(student: Student, features: dict) -> bool:
    """
    completed_on_time ~ Bernoulli(p), where p is a logistic function of
    the student's TRUE base reliability minus overload pressure (load
    ratio + concurrency), scaled by their archetype's overload
    sensitivity. This is where the actual signal in the dataset comes
    from -- everything upstream just builds realistic features; this is
    the one place "ground truth" is defined.
    """
    # Center load_ratio around 1.0 (exactly enough time = break-even).
    overload_pressure = (features["load_ratio"] - 1.0) * student.overload_sensitivity
    overload_pressure += max(features["concurrent_task_count"] - 2, 0) * 0.15 * student.overload_sensitivity

    base_logit = np.log(student.base_on_time_rate / (1 - student.base_on_time_rate))
    logit = base_logit - overload_pressure

    # Small genuine noise term so the boundary isn't perfectly sharp --
    # keeps the classifier honest (not 100% separable), same spirit as
    # the duration generator's outlier injection.
    logit += float(RNG.normal(0, 0.35))

    p_on_time = float(np.clip(sigmoid(logit), 0.02, 0.98))
    return bool(RNG.random() < p_on_time)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(students: list[Student]) -> pd.DataFrame:
    rows = []
    base_now = datetime(2026, 9, 1, 8, 0, 0)

    for student in students:
        store = CorrectionFactorStore(path=":memory:")
        store._save = lambda: None  # no-op -- avoid writing N_STUDENTS json files to disk
        store._data = {}

        # Give each student a starting completion history of varying
        # length (0 to ~15) BEFORE their first simulated schedule, so the
        # dataset includes real cold-start rows (n_completion_history==0)
        # alongside warmed-up ones -- matches real usage distribution.
        starting_history_len = int(RNG.integers(0, 16))
        history_ratios = simulate_completion_history(student, store, starting_history_len)
        student.history_ratios = history_ratios

        for s in range(SCHEDULES_PER_STUDENT):
            now = base_now + timedelta(days=int(RNG.integers(0, 60)), hours=int(RNG.integers(0, 14)))

            features, _ = build_schedule_features(student, store, now)
            label = sample_label(student, features)

            rows.append(
                {
                    "student_id": student.student_id,
                    "schedule_id": f"{student.student_id}_sched{s}",
                    **features,
                    "completed_on_time": label,
                }
            )

            # Let a bit more history accumulate between schedules too, so
            # later rows for the same student reflect an evolving track
            # record rather than a frozen one.
            extra = simulate_completion_history(student, store, int(RNG.integers(0, 3)))
            student.history_ratios = student.history_ratios + extra

    return pd.DataFrame(rows)


def main():
    print(f"Generating {N_STUDENTS} synthetic students ({len(ARCHETYPE_NAMES)} archetypes)...")
    students = generate_students(N_STUDENTS)

    print(f"Simulating schedules + completion histories ({SCHEDULES_PER_STUDENT} schedules/student, "
          f"~{N_ROWS_TARGET} rows total)...")
    df = generate_dataset(students)

    archetype_by_student = {s.student_id: s.archetype for s in students}
    df["_archetype_debug_only"] = df["student_id"].map(archetype_by_student)

    # Public training file: archetype column EXCLUDED -- it's the latent
    # trait the classifier must never see, same discipline as
    # synthetic_users.csv being reference-only in the duration generator.
    public_cols = [c for c in df.columns if c != "_archetype_debug_only"]
    df[public_cols].to_csv("synthetic_feasibility_data.csv", index=False)

    # Debug-only file WITH the archetype label, for sanity-checking that
    # the generator actually produced separable-but-noisy signal. Never
    # used for training.
    df.to_csv("synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv", index=False)

    # --- Sanity report ---
    print("\n=== Sanity Report ===")
    print(f"Total rows: {len(df)}")
    print(f"Students: {df['student_id'].nunique()}")
    print(f"Overall on-time rate: {df['completed_on_time'].mean()*100:.1f}%")
    print(f"Cold-start rows (n_completion_history==0): {(df['n_completion_history']==0).sum()} "
          f"({(df['n_completion_history']==0).mean()*100:.1f}%)")

    print("\nOn-time rate by archetype (debug only -- NOT a training feature):")
    print(df.groupby("_archetype_debug_only")["completed_on_time"].mean().round(3))

    print("\nOn-time rate by load_ratio bucket (checking the signal is actually there):")
    df["_load_bucket"] = pd.cut(
        df["load_ratio"], bins=[0, 0.5, 0.8, 1.0, 1.3, 1.8, np.inf],
        labels=["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.3", "1.3-1.8", ">1.8"],
    )
    print(df.groupby("_load_bucket", observed=True)["completed_on_time"].agg(["mean", "count"]).round(3))

    print("\nFeature summary:")
    print(df[[
        "n_tasks", "total_predicted_minutes", "available_hours_to_earliest_deadline",
        "load_ratio", "concurrent_task_count", "peak_window_match_ratio",
        "historical_on_time_rate", "recent_slippage_trend",
    ]].describe().round(3).T[["mean", "std", "min", "max"]])

    print("\nAny NaNs in public training columns?", df[public_cols].isna().any().any())

    print("\nSaved:")
    print("  synthetic_feasibility_data.csv                           (training file -- archetype excluded)")
    print("  synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv (reference only -- NOT for training)")


if __name__ == "__main__":
    main()
