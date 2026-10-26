"""
FlowState AI — Schedule Feasibility Model (Stage 5)
=======================================================
Loads the trained LogisticRegression classifier + scaler + decision
threshold (train_feasibility_model.py's output) and exposes
predict_feasibility(), matching the shape TODO_backend_ml.md item 2
specifies:

    predict_feasibility(schedule_result, user_id)
        -> {"completion_probability": float, "risk_flagged_task_id": str | None}

Mirrors pipeline.py's own structure on purpose (lazy-loaded module-level
singletons via _get_*(), same docstring conventions) so this reads as
the same codebase, not a bolted-on second system.
"""

import datetime as _dt
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from pipeline import CorrectionFactorStore, _get_correction_store  # reuse the real store

MODEL_PATH = "feasibility_model.joblib"
SCALER_PATH = "feasibility_scaler.joblib"
THRESHOLD_PATH = "feasibility_threshold.joblib"

FEATURE_COLS = [
    "n_tasks",
    "total_predicted_minutes",
    "available_hours_to_earliest_deadline",
    "load_ratio",
    "concurrent_task_count",
    "peak_window_match_ratio",
    "historical_on_time_rate",
    "recent_slippage_trend",
    "n_completion_history",
]

_model = None
_scaler = None
_threshold = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def _get_threshold() -> float:
    global _threshold
    if _threshold is None:
        _threshold = joblib.load(THRESHOLD_PATH)["infeasible_threshold"]
    return _threshold


# ---------------------------------------------------------------------------
# Feature extraction from a real /api/schedule/generate result
# ---------------------------------------------------------------------------
# NOTE: this is where production will differ slightly from the synthetic
# generator. generate_feasibility_data.py had ground-truth access to each
# student's TRUE completion-ratio history (it generated that history
# itself). In production, "historical on-time rate" and "recent slippage
# trend" have to be derived from whatever CorrectionFactorStore actually
# has on file for this user -- which only tracks a factor + n_samples per
# (user, task_type), not a raw per-completion ratio log. So here we
# approximate both stats from the stored factor itself (factor IS the EMA
# of the actual/predicted ratio, so it's a reasonable stand-in), averaged
# across the task types that appear in this schedule. This is a known,
# documented approximation -- flag it if asked "is this exactly how you
# trained it."

def _historical_stats_from_store(
    store: CorrectionFactorStore, user_id: str, task_types: list[str]
) -> tuple[float, float, int]:
    entries = [store.get(user_id, t) for t in set(task_types)]
    entries = [e for e in entries if e.n_samples > 0]

    if not entries:
        return 0.5, 1.0, 0  # cold start -- same neutral prior as training data

    avg_factor = float(np.mean([e.factor for e in entries]))
    total_samples = sum(e.n_samples for e in entries)
    # factor is the EMA of actual/predicted; treat <=1.15x as "on time"
    # (slight slack), matching the generator's own definition exactly.
    historical_on_time_rate = 1.0 if avg_factor <= 1.15 else 0.0
    # With only a stored factor (not a raw ratio log) we can't compute a
    # true "last N" trend -- the factor itself already IS a recency-
    # weighted average, so we use it directly as the trend estimate.
    recent_slippage_trend = avg_factor

    return historical_on_time_rate, recent_slippage_trend, total_samples


def _extract_features(schedule_result: dict, user_id: str, now: _dt.datetime) -> dict:
    """
    schedule_result is exactly what generate_schedule() / the
    /api/schedule/generate endpoint returns: {"task_analysis": [...], "schedule": [...]}.
    """
    task_analysis = schedule_result["task_analysis"]
    if not task_analysis:
        raise ValueError("schedule_result['task_analysis'] is empty")

    store = _get_correction_store()

    total_predicted_minutes = sum(t["predicted_duration_minutes"] for t in task_analysis)
    deadlines = [_dt.datetime.fromisoformat(t["buffer_deadline"]) for t in task_analysis]
    earliest_buffer = min(deadlines)
    available_hours = max((earliest_buffer - now).total_seconds() / 3600, 0.5)
    load_ratio = (total_predicted_minutes / 60) / available_hours

    concurrent_task_count = sum(
        1 for bd in deadlines if bd <= earliest_buffer + _dt.timedelta(hours=12)
    )

    # Same >=4h-runway proxy as app.py's estimate_in_peak_window() /
    # the generator -- kept identical rather than reinvented.
    in_peak_flags = []
    for t, bd in zip(task_analysis, deadlines):
        # task_analysis doesn't carry the raw deadline, only buffer_deadline;
        # approximate original deadline runway from buffer_deadline, which
        # is close enough for this coarse flag (buffer is only 15% of the window).
        hours_remaining = (bd - now).total_seconds() / 3600
        in_peak_flags.append(1 if hours_remaining >= 4 else 0)
    peak_window_match_ratio = float(np.mean(in_peak_flags))

    task_types = [t["task_type"] for t in task_analysis]
    historical_on_time_rate, recent_slippage_trend, n_completion_history = (
        _historical_stats_from_store(store, user_id, task_types)
    )

    return {
        "n_tasks": len(task_analysis),
        "total_predicted_minutes": total_predicted_minutes,
        "available_hours_to_earliest_deadline": available_hours,
        "load_ratio": load_ratio,
        "concurrent_task_count": concurrent_task_count,
        "peak_window_match_ratio": peak_window_match_ratio,
        "historical_on_time_rate": historical_on_time_rate,
        "recent_slippage_trend": recent_slippage_trend,
        "n_completion_history": n_completion_history,
    }


# ---------------------------------------------------------------------------
# risk_flagged_task_id — simplest version per the TODO spec: the task
# with the least slack (buffer_deadline vs. predicted duration), i.e.
# the one most likely to be the actual bottleneck.
# ---------------------------------------------------------------------------

def _identify_risk_flagged_task(task_analysis: list[dict], now: _dt.datetime) -> Optional[str]:
    if not task_analysis:
        return None

    def slack_ratio(t: dict) -> float:
        buffer_deadline = _dt.datetime.fromisoformat(t["buffer_deadline"])
        hours_to_buffer = max((buffer_deadline - now).total_seconds() / 3600, 0.01)
        return hours_to_buffer / max(t["predicted_duration_minutes"] / 60, 0.01)

    return min(task_analysis, key=slack_ratio)["task_id"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def predict_feasibility(schedule_result: dict, user_id: str, now: Optional[_dt.datetime] = None) -> dict:
    """
    Stage 5: call this from /api/schedule/generate's handler, right after
    generate_schedule() succeeds, same pattern as analyze_task() is
    called per-task in Stage 1+2.

    Returns: {"completion_probability": float, "risk_flagged_task_id": str | None}
    """
    now = now or _dt.datetime.now()

    features = _extract_features(schedule_result, user_id, now)
    # DataFrame with named columns -- scaler/model were fit on FEATURE_COLS
    # by name (same discipline as pipeline.py's _build_feature_row()), so
    # this avoids a sklearn "X does not have valid feature names" warning
    # and keeps column order unambiguous regardless of dict ordering.
    row = pd.DataFrame([features], columns=FEATURE_COLS)

    scaler = _get_scaler()
    model = _get_model()
    row_scaled = scaler.transform(row)

    completion_probability = float(model.predict_proba(row_scaled)[0, 1])  # P(on_time)

    threshold = _get_threshold()
    p_infeasible = 1 - completion_probability
    is_at_risk = p_infeasible >= threshold

    risk_flagged_task_id = (
        _identify_risk_flagged_task(schedule_result["task_analysis"], now) if is_at_risk else None
    )

    return {
        "completion_probability": round(completion_probability, 3),
        "risk_flagged_task_id": risk_flagged_task_id,
    }
