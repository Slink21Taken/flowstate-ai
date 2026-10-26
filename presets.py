"""
FlowState AI — Preset Persistence
=====================================
Backend-saved alternative to localStorage-only session persistence (see
TODO_backend_ml.md item 3). Decision made: judges may test on a machine/
browser other than the one the session was created on, so a save that
only survives in one browser's localStorage isn't good enough.

Stores the last parsedTasks / taskAnalysis / schedule blob per user_id,
JSON-file-backed the same way correction_factors.json already works via
CorrectionFactorStore in pipeline.py. One preset per user_id -- a new
save overwrites the previous one (not a history list). This matches what
the feature is actually for here: "resume where I left off," not
"browse past sessions."

Swap _load/_save for a real DB call later without touching the
get/save/delete logic at all -- same escape hatch CorrectionFactorStore
already has.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

PRESET_STORE_PATH = "presets.json"


@dataclass
class PresetEntry:
    parsed_tasks: list = field(default_factory=list)
    task_analysis: list = field(default_factory=list)
    schedule: list = field(default_factory=list)
    feasibility: Optional[dict] = None
    saved_at: Optional[str] = None


class PresetStore:
    """
    Tracks the single most recent saved preset per user_id.

    Persisted as JSON for the hackathon demo, same pattern as
    CorrectionFactorStore in pipeline.py -- _load/_save are the only
    methods that would need to change to move this to a real DB.
    """

    def __init__(self, path: str = PRESET_STORE_PATH):
        self.path = path
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, user_id: str) -> Optional[PresetEntry]:
        """Returns None if this user has never saved a preset (distinct
        from an empty-but-saved preset, which returns a PresetEntry with
        empty lists)."""
        raw = self._data.get(user_id)
        if raw is None:
            return None
        return PresetEntry(**raw)

    def save(
        self,
        user_id: str,
        parsed_tasks: list,
        task_analysis: list,
        schedule: list,
        feasibility: Optional[dict] = None,
    ) -> PresetEntry:
        """Overwrites any existing preset for this user_id -- latest wins."""
        entry = PresetEntry(
            parsed_tasks=parsed_tasks,
            task_analysis=task_analysis,
            schedule=schedule,
            feasibility=feasibility,
            saved_at=datetime.now(timezone.utc).isoformat(),
        )
        self._data[user_id] = {
            "parsed_tasks": entry.parsed_tasks,
            "task_analysis": entry.task_analysis,
            "schedule": entry.schedule,
            "feasibility": entry.feasibility,
            "saved_at": entry.saved_at,
        }
        self._save()
        return entry

    def delete(self, user_id: str) -> bool:
        """Returns True if a preset existed and was removed, False if
        there was nothing to delete. Used by the frontend's planned
        "Clear saved session" control."""
        if user_id in self._data:
            del self._data[user_id]
            self._save()
            return True
        return False


# ---------------------------------------------------------------------------
# Module-level singleton, same lazy-init pattern as pipeline.py's
# _get_correction_store()
# ---------------------------------------------------------------------------

_preset_store: Optional[PresetStore] = None


def _get_preset_store() -> PresetStore:
    global _preset_store
    if _preset_store is None:
        _preset_store = PresetStore()
    return _preset_store


def get_preset(user_id: str) -> Optional[dict]:
    """Returns the saved preset as a plain dict (API-response shape), or
    None if nothing's saved for this user_id."""
    entry = _get_preset_store().get(user_id)
    if entry is None:
        return None
    return {
        "parsed_tasks": entry.parsed_tasks,
        "task_analysis": entry.task_analysis,
        "schedule": entry.schedule,
        "feasibility": entry.feasibility,
        "saved_at": entry.saved_at,
    }


def save_preset(
    user_id: str,
    parsed_tasks: list,
    task_analysis: list,
    schedule: list,
    feasibility: Optional[dict] = None,
) -> dict:
    entry = _get_preset_store().save(
        user_id=user_id,
        parsed_tasks=parsed_tasks,
        task_analysis=task_analysis,
        schedule=schedule,
        feasibility=feasibility,
    )
    return {
        "parsed_tasks": entry.parsed_tasks,
        "task_analysis": entry.task_analysis,
        "schedule": entry.schedule,
        "feasibility": entry.feasibility,
        "saved_at": entry.saved_at,
    }


def delete_preset(user_id: str) -> bool:
    return _get_preset_store().delete(user_id)
