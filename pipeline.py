"""
FlowState AI — Inference Pipeline (Stage 1+2 combined) + Correction Factor (Stage 4)
========================================================================================
This is the code that sits behind /api/schedule/generate and /api/tasks/complete.

Two pieces:
  1. analyze_task()        -> text/metadata in, predicted_duration_minutes out.
                               Combines feature extraction + CatBoost prediction +
                               per-user correction factor application, in one call,
                               per the "combine Stage 1+2" decision made earlier.
  2. CorrectionFactorStore -> tracks each user's per-task-type learned adjustment
                               and updates it after every completed task.

IMPORTANT — feature parity with training:
analyze_task() must build EXACTLY the same features, in the same way, as
train_model.py's engineer_features(). If these drift apart, the model will
silently make garbage predictions because it's seeing differently-shaped
input than it was trained on. _build_feature_row() below is deliberately
written as a close mirror of train_model.py's engineer_features() — if you
change one, change the other.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

MODEL_PATH = "catboost_model.cbm"
CORRECTION_STORE_PATH = "correction_factors.json"

FEATURE_COLS = [
    "task_type",
    "base_difficulty",
    "difficulty_squared",
    "deadline_hours_remaining",
    "log_deadline_hours",
    "is_crunch",
    "is_far_out",
    "in_peak_window",
]
CATEGORICAL_COLS = ["task_type"]
TASK_TYPES = ["coding", "essay", "reading", "math", "creative"]

# Correction factor tuning
EMA_ALPHA = 0.3          # weight given to the newest observation
FACTOR_MIN = 0.4         # clamp bounds -- stop one outlier completion from
FACTOR_MAX = 2.5         # sending future predictions to absurd values
MIN_SAMPLES_FOR_FULL_TRUST = 3   # see _blend_factor() below


# ---------------------------------------------------------------------------
# Feature building — MUST mirror train_model.py's engineer_features()
# ---------------------------------------------------------------------------

def _build_feature_row(
    task_type: str,
    base_difficulty: int,
    deadline_hours_remaining: float,
    in_peak_window: bool,
) -> pd.DataFrame:
    """Builds a single-row DataFrame in the exact shape the model expects."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unknown task_type '{task_type}', expected one of {TASK_TYPES}")
    if not (1 <= base_difficulty <= 10):
        raise ValueError(f"base_difficulty must be 1-10, got {base_difficulty}")
    if deadline_hours_remaining <= 0:
        raise ValueError(f"deadline_hours_remaining must be positive, got {deadline_hours_remaining}")

    row = {
        "task_type": task_type,
        "base_difficulty": base_difficulty,
        "difficulty_squared": base_difficulty ** 2,
        "deadline_hours_remaining": deadline_hours_remaining,
        "log_deadline_hours": np.log1p(deadline_hours_remaining),
        "is_crunch": int(deadline_hours_remaining < 6),
        "is_far_out": int(deadline_hours_remaining > 120),
        "in_peak_window": int(in_peak_window),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ---------------------------------------------------------------------------
# Correction Factor Store (Stage 4)
# ---------------------------------------------------------------------------

@dataclass
class CorrectionEntry:
    factor: float = 1.0
    n_samples: int = 0
    last_updated: Optional[str] = None


class CorrectionFactorStore:
    """
    Tracks per-(user_id, task_type) correction factors.

    Persisted as JSON for the hackathon demo -- swap _load/_save for a real
    DB call later without touching the update math at all.
    """

    def __init__(self, path: str = CORRECTION_STORE_PATH):
        self.path = path
        self._data: dict[str, dict[str, dict]] = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, user_id: str, task_type: str) -> CorrectionEntry:
        user_entry = self._data.get(user_id, {})
        raw = user_entry.get(task_type)
        if raw is None:
            return CorrectionEntry()
        return CorrectionEntry(**raw)

    def update(
        self, user_id: str, task_type: str, predicted_minutes: float, actual_minutes: float
    ) -> CorrectionEntry:
        """
        EMA update on the actual/predicted RATIO (not raw minute delta --
        ratios compose cleanly with the model's multiplicative correction,
        whereas a flat minute delta means very different things for a
        30-min reading task vs. a 150-min coding task).

        new_factor = old_factor * (1 - alpha) + observed_ratio * alpha

        Clamped to [FACTOR_MIN, FACTOR_MAX] so a single wild outlier
        completion (which the training data deliberately includes ~5-6%
        of) can't send future predictions for that user/type to an
        absurd place in one update.
        """
        if predicted_minutes <= 0 or not math.isfinite(predicted_minutes):
            raise ValueError("predicted_minutes must be positive")
        if actual_minutes <= 0 or not math.isfinite(actual_minutes):
            raise ValueError("actual_minutes must be positive")

        current = self.get(user_id, task_type)
        observed_ratio = actual_minutes / predicted_minutes

        new_factor = current.factor * (1 - EMA_ALPHA) + observed_ratio * EMA_ALPHA
        new_factor = float(np.clip(new_factor, FACTOR_MIN, FACTOR_MAX))

        updated = CorrectionEntry(
            factor=new_factor,
            n_samples=current.n_samples + 1,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self._data.setdefault(user_id, {})[task_type] = {
            "factor": updated.factor,
            "n_samples": updated.n_samples,
            "last_updated": updated.last_updated,
        }
        self._save()
        return updated

    def effective_factor(self, user_id: str, task_type: str) -> float:
        """
        Blends the learned factor toward 1.0 (no adjustment) when there
        are very few samples, so one early completion doesn't overcorrect
        with full confidence. Linearly ramps to full trust in the learned
        factor by MIN_SAMPLES_FOR_FULL_TRUST completions.

        This matters because EMA with alpha=0.3 on a single sample already
        moves the stored factor 30% of the way to that one observation --
        this blend additionally tempers how much of THAT we actually
        apply at inference time until there's more evidence.
        """
        entry = self.get(user_id, task_type)
        if entry.n_samples == 0:
            return 1.0
        trust = min(entry.n_samples / MIN_SAMPLES_FOR_FULL_TRUST, 1.0)
        return 1.0 * (1 - trust) + entry.factor * trust


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class EffortPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        self.cat_idx = [FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]

    def predict_population_duration(
        self,
        task_type: str,
        base_difficulty: int,
        deadline_hours_remaining: float,
        in_peak_window: bool,
    ) -> float:
        """Raw model output -- population-level prediction, NOT personalized."""
        row = _build_feature_row(task_type, base_difficulty, deadline_hours_remaining, in_peak_window)
        pool = Pool(row, cat_features=self.cat_idx)
        pred = float(self.model.predict(pool)[0])
        return max(pred, 5.0)  # floor at 5 min -- avoid degenerate near-zero predictions


# ---------------------------------------------------------------------------
# Stage 1+2 combined: analyze_task()
# ---------------------------------------------------------------------------

_predictor: Optional[EffortPredictor] = None
_correction_store: Optional[CorrectionFactorStore] = None


def _get_predictor() -> EffortPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EffortPredictor()
    return _predictor


def _get_correction_store() -> CorrectionFactorStore:
    global _correction_store
    if _correction_store is None:
        _correction_store = CorrectionFactorStore()
    return _correction_store


def explain_prediction(
    task_type: str,
    correction: float,
    population_duration: float,
    final_duration: float,
    is_crunch: bool,
    n_samples: int,
) -> str:
    """
    Pure string templating -- no Groq call. See TODO_backend_ml.md item 1:
    the inputs here are already computed by analyze_task(), so turning
    them into one sentence is formatting, not a judgment call, and should
    stay instant/always-available rather than adding a new 502 failure
    mode for something that doesn't need reasoning.
    """
    if n_samples == 0:
        return (
            f"First time predicting a {task_type} task for you — "
            f"using the general model until we have your history."
        )

    pct = abs(round((correction - 1.0) * 100))
    direction = "longer" if correction > 1.0 else "shorter"
    speed_word = "slower" if correction > 1.0 else "faster"

    sentence = (
        f"{pct}% {direction} than our baseline guess, because you've "
        f"historically run {speed_word} on {task_type} tasks"
    )
    if is_crunch and correction > 1.0:
        sentence += ", especially under tight deadlines"
    sentence += "."
    return sentence


def analyze_task(
    task_id: str,
    task_type: str,
    base_difficulty: int,
    deadline_hours_remaining: float,
    in_peak_window: bool,
    user_id: str,
) -> dict:
    """
    Stage 1+2 combined, per the architecture decision: feature extraction
    (Step A/B) and CatBoost prediction (Step C) run in-process as one
    function, then the per-user correction factor (Step D) is applied.

    NOTE: task_type/base_difficulty here are assumed ALREADY EXTRACTED
    from raw text (that's the separate LLM-prompt step upstream, per the
    "fake the NLP layer cheaply" decision -- this function starts from
    structured fields, matching what /api/tasks/parse already returns).
    """
    predictor = _get_predictor()
    store = _get_correction_store()

    population_duration = predictor.predict_population_duration(
        task_type, base_difficulty, deadline_hours_remaining, in_peak_window
    )

    correction = store.effective_factor(user_id, task_type)
    final_duration = population_duration * correction

    # n_samples drives the "first time" vs. comparative explanation branch;
    # store.get() (not effective_factor()) is the one call that exposes it.
    n_samples = store.get(user_id, task_type).n_samples
    is_crunch = deadline_hours_remaining < 6  # same threshold _build_feature_row() uses

    explanation = explain_prediction(
        task_type=task_type,
        correction=correction,
        population_duration=population_duration,
        final_duration=final_duration,
        is_crunch=is_crunch,
        n_samples=n_samples,
    )

    return {
        "task_id": task_id,
        "task_type": task_type,
        "difficulty": base_difficulty,
        "population_predicted_duration_minutes": round(population_duration, 1),
        "correction_factor_applied": round(correction, 3),
        "predicted_duration_minutes": round(final_duration, 1),
        "explanation": explanation,
    }


def complete_task(
    user_id: str,
    task_type: str,
    predicted_duration_minutes: float,
    actual_duration_minutes: float,
) -> dict:
    """
    Stage 4: call this from /api/tasks/complete after a user marks a task
    done. Updates the stored correction factor and returns the new
    effective value so the API/UI can show the "model adjusted" moment.
    """
    store = _get_correction_store()
    updated = store.update(user_id, task_type, predicted_duration_minutes, actual_duration_minutes)
    new_effective = store.effective_factor(user_id, task_type)

    pct_change = (new_effective - 1.0) * 100
    direction = "longer" if pct_change > 0 else "shorter"

    return {
        "updated_correction_factor": round(updated.factor, 3),
        "effective_factor": round(new_effective, 3),
        "n_samples": updated.n_samples,
        "message": f"Future {task_type} predictions adjusted {abs(pct_change):.0f}% {direction}",
    }
