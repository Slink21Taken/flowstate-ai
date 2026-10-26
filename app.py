"""
FlowState AI — Flask API
============================
Wires Stages 0 through 4 into the 3 endpoints from the original API
contract, plus a 4th endpoint for image-based ingestion (vision_parser.py),
plus Stage 5 (schedule feasibility) added into /api/schedule/generate's
response, plus Stage 6 (preset persistence) via /api/presets.

Endpoints:
  POST   /api/tasks/parse        Stage 0+1 (Groq, text)   -> structured tasks
  POST   /api/tasks/parse/image  Stage 0+1 (Groq, vision)  -> structured tasks
  POST   /api/schedule/generate  Stage 1+2+3+5             -> predictions + schedule + feasibility
  POST   /api/tasks/complete     Stage 4                   -> correction factor update
  POST   /api/presets            Stage 6                   -> save a session (overwrites)
  GET    /api/presets/<user_id>  Stage 6                   -> load a saved session
  DELETE /api/presets/<user_id>  Stage 6                   -> clear a saved session

Run:
    export GROQ_API_KEY="your-key-here"
    python3 app.py
    (serves on http://localhost:5000)
"""


from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS
from parse import parse_tasks, ParsedTask
from vision_parser import parse_tasks_from_image_bytes, _SUPPORTED_MIME_TYPES
from pipeline import analyze_task, complete_task
from scheduler import generate_schedule, TaskForScheduling, PEAK_WINDOWS
from feasibility_model import predict_feasibility  # Stage 5
from presets import get_preset, save_preset, delete_preset  # Stage 6

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Shared helper -- mirrors run_pipeline.py's estimate_in_peak_window()
# ---------------------------------------------------------------------------

def estimate_in_peak_window(task_deadline: datetime, now: datetime) -> bool:
    """
    See run_pipeline.py for the full rationale. Short version: at
    prediction time no schedule slot exists yet, so this is a proxy --
    "is there enough runway before the deadline that doing this task
    during the user's peak window is plausible" -- not a lookup against
    the final schedule (which runs after this, in Stage 3).
    """
    hours_remaining = (task_deadline - now).total_seconds() / 3600
    return hours_remaining >= 4


def error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status



def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_aware_utc(dt: datetime) -> datetime:
    """Normalize any datetime to UTC-aware. Naive datetimes are assumed
    to already be in UTC (the convention used everywhere else in this
    codebase before timezone-aware input existed)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# ---------------------------------------------------------------------------
# POST /api/tasks/parse
# ---------------------------------------------------------------------------
# Request:  {"raw_text": "..."}
# Response: {"tasks": [{"task_id", "title", "task_type", "deadline", "base_difficulty"}, ...]}

@app.route("/api/tasks/parse", methods=["POST"])
def tasks_parse():
    body = request.get_json(silent=True)
    if not body or "raw_text" not in body:
        return error_response("Request body must include 'raw_text'")

    raw_text = body["raw_text"]
    if not isinstance(raw_text, str) or not raw_text.strip():
        return error_response("'raw_text' must be a non-empty string")

    now = _now_utc()

    try:
        parsed_tasks: list[ParsedTask] = parse_tasks(raw_text, now=now)
    except RuntimeError as e:
        # Covers both missing GROQ_API_KEY and Groq call failure after retry
        return error_response(str(e), status=502)

    return jsonify({"tasks": [t.to_dict() for t in parsed_tasks]})


# ---------------------------------------------------------------------------
# POST /api/tasks/parse/image
# ---------------------------------------------------------------------------
# Request:  multipart/form-data with an "image" file field
#           (screenshot of a syllabus, Canvas page, planner, etc.)
# Response: same shape as /api/tasks/parse -- {"tasks": [...]}
#
# Kept as a separate endpoint rather than overloading /api/tasks/parse so
# the original text contract is untouched (additive, same spirit as the
# rest of this file's design). Frontend picks whichever endpoint matches
# the input it has (typed/pasted text vs. an uploaded/pasted image).

@app.route("/api/tasks/parse/image", methods=["POST"])
def tasks_parse_image():
    if "image" not in request.files:
        return error_response("Request must include an 'image' file field (multipart/form-data)")

    file = request.files["image"]
    if file.filename == "":
        return error_response("No image file selected")

    image_bytes = file.read()
    if not image_bytes:
        return error_response("Uploaded image is empty")

    # Prefer the browser-reported content type; fall back to sniffing from
    # the filename extension if that's missing/generic, since some
    # clipboard-paste flows don't set a reliable mimetype.
    mime_type = file.mimetype
    if mime_type not in _SUPPORTED_MIME_TYPES:
        import mimetypes as _mimetypes
        guessed, _ = _mimetypes.guess_type(file.filename or "")
        mime_type = guessed

    if mime_type not in _SUPPORTED_MIME_TYPES:
        return error_response(
            f"Unsupported image type '{file.mimetype}'. "
            f"Supported: {sorted(_SUPPORTED_MIME_TYPES)}"
        )

    now = _now_utc()

    try:
        parsed_tasks: list[ParsedTask] = parse_tasks_from_image_bytes(
            image_bytes, mime_type=mime_type, now=now
        )
    except ValueError as e:
        # Bad/oversized image -- client error, not a server/API failure
        return error_response(str(e), status=400)
    except RuntimeError as e:
        # Missing GROQ_API_KEY or Groq call failure after retry
        return error_response(str(e), status=502)

    return jsonify({"tasks": [t.to_dict() for t in parsed_tasks]})


# ---------------------------------------------------------------------------
# POST /api/schedule/generate
# ---------------------------------------------------------------------------
# Request:  {"tasks": [{"task_id", "title", "task_type", "deadline", "base_difficulty"}, ...],
#            "user_id": "...", "peak_window": "morning"|"evening"}
# Response: {"task_analysis": [...], "schedule": [...], "feasibility": {...}}
#
# "feasibility" is Stage 5, added after generate_schedule() succeeds, per
# TODO_backend_ml.md item 2's wiring instruction: call predict_feasibility()
# on the schedule_result, add its output as a new top-level key. If the
# feasibility model (or its joblib artifacts) isn't available for any
# reason, the endpoint still returns task_analysis + schedule -- a broken
# or missing Stage 5 model must NEVER take down Stage 1+2+3, since the
# scheduler was shipping correctly before this model existed.

@app.route("/api/schedule/generate", methods=["POST"])
def schedule_generate():
    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be JSON")

    required = ["tasks", "user_id", "peak_window"]
    missing = [f for f in required if f not in body]
    if missing:
        return error_response(f"Missing required field(s): {', '.join(missing)}")

    raw_tasks = body["tasks"]
    user_id = body["user_id"]
    peak_window = body["peak_window"]

    if not isinstance(raw_tasks, list) or not raw_tasks:
        return error_response("'tasks' must be a non-empty array")
    if peak_window not in PEAK_WINDOWS:
        return error_response(f"'peak_window' must be one of {list(PEAK_WINDOWS)}")

    now = _now_utc()

    # --- Parse incoming task dicts into typed objects, validating as we go ---
    parsed = []
    for i, t in enumerate(raw_tasks):
        for field in ("task_id", "title", "task_type", "deadline", "base_difficulty"):
            if field not in t:
                return error_response(f"tasks[{i}] missing required field '{field}'")
        try:
            deadline = _to_aware_utc(datetime.fromisoformat(t["deadline"]))
        except (ValueError, TypeError):
            return error_response(f"tasks[{i}].deadline must be a valid ISO datetime string")

        try:
            difficulty = int(t["base_difficulty"])
        except (TypeError, ValueError):
            return error_response(f"tasks[{i}].base_difficulty must be an integer")

        parsed.append(
            {
                "task_id": t["task_id"],
                "title": t["title"],
                "task_type": t["task_type"],
                "deadline": deadline,
                "base_difficulty": difficulty,
            }
        )

    # --- Stage 1+2: predict duration per task (population model + correction factor) ---
    schedule_input = []
    for t in parsed:
        deadline_hours_remaining = max((t["deadline"] - now).total_seconds() / 3600, 0.5)
        in_peak = estimate_in_peak_window(t["deadline"], now)

        try:
            prediction = analyze_task(
                task_id=t["task_id"],
                task_type=t["task_type"],
                base_difficulty=t["base_difficulty"],
                deadline_hours_remaining=deadline_hours_remaining,
                in_peak_window=in_peak,
                user_id=user_id,
            )
        except ValueError as e:
            return error_response(f"tasks[{t['task_id']}]: {e}")

        schedule_input.append(
            TaskForScheduling(
                task_id=t["task_id"],
                title=t["title"],
                task_type=t["task_type"],
                difficulty=t["base_difficulty"],
                predicted_duration_minutes=prediction["predicted_duration_minutes"],
                deadline=t["deadline"],
                explanation=prediction.get("explanation"),
            )
        )

    # --- Stage 3: schedule ---
    try:
        schedule_result = generate_schedule(schedule_input, peak_window=peak_window, now=now)
    except ValueError as e:
        return error_response(str(e))

    # --- Stage 5: feasibility (additive -- never fails the whole request) ---
    try:
        schedule_result["feasibility"] = predict_feasibility(
            schedule_result, user_id=user_id, now=now
        )
    except Exception as e:
        # Deliberately broad: ANY feasibility-model failure (missing
        # joblib files, a malformed row, etc.) degrades gracefully rather
        # than taking down a request that Stage 1+2+3 already answered
        # correctly. Surfaced as null + a visible reason rather than
        # silently omitted, so a flaky model doesn't look like "no risk
        # detected" to the frontend/judges.
        schedule_result["feasibility"] = {
            "completion_probability": None,
            "risk_flagged_task_id": None,
            "error": f"feasibility model unavailable: {e}",
        }

    return jsonify(schedule_result)


# ---------------------------------------------------------------------------
# POST /api/tasks/complete
# ---------------------------------------------------------------------------
# Request:  {"user_id", "task_id", "task_type", "predicted_duration_minutes",
#            "actual_duration_minutes", "fatigue_self_report"}
# Response: {"updated_correction_factor", "message", "effective_factor", "n_samples"}

@app.route("/api/tasks/complete", methods=["POST"])
def tasks_complete():
    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be JSON")

    required = ["user_id", "task_id", "task_type", "predicted_duration_minutes", "actual_duration_minutes"]
    missing = [f for f in required if f not in body]
    if missing:
        return error_response(f"Missing required field(s): {', '.join(missing)}")

    try:
        predicted = float(body["predicted_duration_minutes"])
        actual = float(body["actual_duration_minutes"])
    except (TypeError, ValueError):
        return error_response("predicted_duration_minutes and actual_duration_minutes must be numbers")

    if body["task_type"] not in ("coding", "essay", "reading", "math", "creative"):
        return error_response("Invalid task_type")

    try:
        result = complete_task(
            user_id=body["user_id"],
            task_type=body["task_type"],
            predicted_duration_minutes=predicted,
            actual_duration_minutes=actual,
        )
    except ValueError as e:
        return error_response(str(e))

    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /api/presets
# ---------------------------------------------------------------------------
# Request:  {"user_id": "...", "parsed_tasks": [...], "task_analysis": [...],
#            "schedule": [...], "feasibility": {...} }
#           ("feasibility" is optional -- a preset saved before Stage 5
#           existed, or saved from a response where feasibility degraded
#           to null, shouldn't be rejected for missing it.)
# Response: {"saved_at", "message"}
#
# Overwrites any existing preset for this user_id -- one saved session per
# user, latest wins (decision made over keeping history: this feature is
# "resume where I left off," not "browse past sessions").

@app.route("/api/presets", methods=["POST"])
def presets_save():
    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be JSON")

    required = ["user_id", "parsed_tasks", "task_analysis", "schedule"]
    missing = [f for f in required if f not in body]
    if missing:
        return error_response(f"Missing required field(s): {', '.join(missing)}")

    user_id = body["user_id"]
    if not isinstance(user_id, str) or not user_id.strip():
        return error_response("'user_id' must be a non-empty string")

    for field_name in ("parsed_tasks", "task_analysis", "schedule"):
        if not isinstance(body[field_name], list):
            return error_response(f"'{field_name}' must be an array")

    feasibility = body.get("feasibility")
    if feasibility is not None and not isinstance(feasibility, dict):
        return error_response("'feasibility' must be an object if provided")

    entry = save_preset(
        user_id=user_id,
        parsed_tasks=body["parsed_tasks"],
        task_analysis=body["task_analysis"],
        schedule=body["schedule"],
        feasibility=feasibility,
    )

    return jsonify({"saved_at": entry["saved_at"], "message": "Preset saved"})


# ---------------------------------------------------------------------------
# GET /api/presets/<user_id>
# ---------------------------------------------------------------------------
# Response: the saved preset blob, or 404 if this user has never saved one.
# A 404 here is a normal, expected state (first-ever visit) -- the frontend
# should treat it as "nothing to restore," not as an error to surface.

@app.route("/api/presets/<user_id>", methods=["GET"])
def presets_get(user_id):
    if not user_id.strip():
        return error_response("'user_id' must be a non-empty string")

    preset = get_preset(user_id)
    if preset is None:
        return error_response(f"No saved preset for user_id '{user_id}'", status=404)

    return jsonify(preset)


# ---------------------------------------------------------------------------
# DELETE /api/presets/<user_id>
# ---------------------------------------------------------------------------
# Response: {"deleted": bool, "message"}
# Backs the frontend's planned "Clear saved session" control (TODO_frontend_ui.md).
# Deleting a preset that doesn't exist is NOT an error -- idempotent by design,
# since "make sure nothing is saved" is the actual intent either way.

@app.route("/api/presets/<user_id>", methods=["DELETE"])
def presets_delete(user_id):
    if not user_id.strip():
        return error_response("'user_id' must be a non-empty string")

    deleted = delete_preset(user_id)
    message = "Preset deleted" if deleted else "No preset existed for this user_id"
    return jsonify({"deleted": deleted, "message": message})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
