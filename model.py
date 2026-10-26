"""
FlowState AI — Feasibility Model Smoke Test
===============================================
Five scenarios run through the REAL pipeline: scheduler.generate_schedule()
produces an actual schedule_result dict (same shape /api/schedule/generate
returns), then feasibility_model.predict_feasibility() scores it -- same
call pattern pipeline.py's own __main__ block uses for analyze_task().

Covers: cold start, a light/easy schedule, a genuinely overloaded one,
and the before/after effect of a student's completion history.
"""

import json
from datetime import datetime, timedelta

from scheduler import TaskForScheduling, generate_schedule
from pipeline import complete_task, _get_correction_store
from feasibility_model import predict_feasibility

NOW = datetime(2026, 10, 26, 8, 0, 0)  # Monday 8am, fixed for reproducibility


def run_scenario(label: str, tasks: list[TaskForScheduling], user_id: str, peak_window: str = "morning"):
    schedule_result = generate_schedule(tasks, peak_window=peak_window, now=NOW)
    feasibility = predict_feasibility(schedule_result, user_id=user_id, now=NOW)

    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {label}")
    print(f"{'=' * 70}")
    print(f"user_id={user_id}  tasks={len(tasks)}  peak_window={peak_window}")
    for t in schedule_result["task_analysis"]:
        print(f"  - {t['title']!r} ({t['task_type']}, diff={t['difficulty']}) "
              f"-> {t['predicted_duration_minutes']}min, buffer_deadline={t['buffer_deadline']}")
    print(f"\n  completion_probability = {feasibility['completion_probability']}")
    print(f"  risk_flagged_task_id   = {feasibility['risk_flagged_task_id']}")
    return feasibility


# ---------------------------------------------------------------------------
# Scenario 1: Cold start -- brand-new student, light schedule
# ---------------------------------------------------------------------------
# Expect: historical features fall back to the neutral prior (0.5 / 1.0),
# so the score leans almost entirely on load_ratio/concurrency, which are
# low here -- should read as reasonably feasible.

cold_start_tasks = [
    TaskForScheduling(
        task_id="c1", title="Read chapter 5", task_type="reading", difficulty=3,
        predicted_duration_minutes=40, deadline=NOW + timedelta(days=2),
    ),
    TaskForScheduling(
        task_id="c2", title="Math worksheet", task_type="math", difficulty=4,
        predicted_duration_minutes=35, deadline=NOW + timedelta(days=3),
    ),
]
run_scenario("Cold start -- brand new student, light load", cold_start_tasks, user_id="brand_new_student")


# ---------------------------------------------------------------------------
# Scenario 2: Light, spread-out schedule -- should score HIGH
# ---------------------------------------------------------------------------

light_tasks = [
    TaskForScheduling(
        task_id="l1", title="Reading response", task_type="reading", difficulty=2,
        predicted_duration_minutes=30, deadline=NOW + timedelta(days=5),
    ),
    TaskForScheduling(
        task_id="l2", title="Short essay outline", task_type="essay", difficulty=3,
        predicted_duration_minutes=45, deadline=NOW + timedelta(days=6),
    ),
]
run_scenario("Light, spread-out schedule", light_tasks, user_id="relaxed_student")


# ---------------------------------------------------------------------------
# Scenario 3: Genuinely overloaded -- many hard tasks, tight shared deadline
# ---------------------------------------------------------------------------
# Expect: high load_ratio + high concurrent_task_count -> LOW completion
# probability, with risk_flagged_task_id populated.

overloaded_tasks = [
    TaskForScheduling(
        task_id="o1", title="Final project code", task_type="coding", difficulty=9,
        predicted_duration_minutes=180, deadline=NOW + timedelta(hours=14),
    ),
    TaskForScheduling(
        task_id="o2", title="Essay draft", task_type="essay", difficulty=8,
        predicted_duration_minutes=120, deadline=NOW + timedelta(hours=16),
    ),
    TaskForScheduling(
        task_id="o3", title="Math problem set", task_type="math", difficulty=9,
        predicted_duration_minutes=150, deadline=NOW + timedelta(hours=18),
    ),
    TaskForScheduling(
        task_id="o4", title="Reading quiz prep", task_type="reading", difficulty=6,
        predicted_duration_minutes=60, deadline=NOW + timedelta(hours=15),
    ),
]
run_scenario("Overloaded -- 4 hard tasks crammed into ~18h", overloaded_tasks, user_id="overloaded_student")


# ---------------------------------------------------------------------------
# Scenario 4 + 5: Same schedule, same student, BEFORE and AFTER a real
# completion history is recorded -- demonstrates the model sharpening its
# read on a specific student as Stage 4 (CorrectionFactorStore) accumulates
# real data, even though the classifier's own weights never change.
# ---------------------------------------------------------------------------

history_demo_tasks = [
    TaskForScheduling(
        task_id="h1", title="Coding assignment", task_type="coding", difficulty=7,
        predicted_duration_minutes=100, deadline=NOW + timedelta(hours=20),
    ),
    TaskForScheduling(
        task_id="h2", title="Math review", task_type="math", difficulty=6,
        predicted_duration_minutes=70, deadline=NOW + timedelta(hours=22),
    ),
]

before = run_scenario(
    "Same schedule, BEFORE completion history exists", history_demo_tasks, user_id="tracked_student"
)

# Simulate this student having consistently run ~40% over their predicted
# time on past coding/math tasks -- same complete_task() call
# /api/tasks/complete triggers in production.
store = _get_correction_store()
for i in range(5):
    complete_task(
        user_id="tracked_student", task_type="coding",
        predicted_duration_minutes=100.0, actual_duration_minutes=140.0,
    )
    complete_task(
        user_id="tracked_student", task_type="math",
        predicted_duration_minutes=70.0, actual_duration_minutes=98.0,
    )

after = run_scenario(
    "Same schedule, AFTER 5 completions showing chronic ~40% overrun",
    history_demo_tasks, user_id="tracked_student",
)

print(f"\n{'=' * 70}")
print("SUMMARY: effect of completion history on the SAME schedule")
print(f"{'=' * 70}")
print(f"  Before history: completion_probability = {before['completion_probability']}")
print(f"  After history:  completion_probability = {after['completion_probability']}")
print(f"  Delta: {round(after['completion_probability'] - before['completion_probability'], 3)}")
print("\n(Model weights are identical in both calls -- the drop reflects real, accumulated")
print(" per-student history flowing through CorrectionFactorStore into the input features,")
print(" not the classifier itself learning anything new.)")

# cleanup demo state so repeated runs start fresh
import os
if os.path.exists("correction_factors.json"):
    os.remove("correction_factors.json")
    print("\n(cleaned up correction_factors.json for repeatable demo runs)")
