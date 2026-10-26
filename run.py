"""
FlowState AI — Full Pipeline CLI
====================================
Wires together all four stages for end-to-end manual testing:

  Stage 0+1 (parse_tasks.py)  : raw text -> structured tasks via Groq
  Stage 0+1 (vision_parser.py): image -> structured tasks via Groq vision
  Stage 1+2 (pipeline.py)     : task features -> predicted duration
                                  (population model + correction factor)
  Stage 3   (scheduler.py)    : predictions -> time-blocked schedule
  Stage 4   (pipeline.py)     : (not exercised here -- that's the
                                  /api/tasks/complete path; this script
                                  only runs the forward prediction+schedule
                                  flow)

Usage:
    export GROQ_API_KEY="your-key-here"
    python3 run.py
    (then type/paste your task list, Ctrl-D or empty line to submit)

    or non-interactively, text:
    python3 run.py --text "Essay due Friday. Math pset due Monday." --user alice --peak morning

    or non-interactively, image:
    python3 run.py --image screenshot.png --user alice --peak morning

Fields that parse_tasks.py / vision_parser.py deliberately do NOT compute
(since they depend on user profile / "now", not on the raw text or image)
are computed HERE:
  - deadline_hours_remaining = (parsed deadline - now)
  - in_peak_window            = whether a task's scheduled slot falls in
                                 the user's declared peak window -- since
                                 at prediction time no slot is assigned
                                 yet, this script approximates it as
                                 "is there enough runway before the
                                 deadline for the peak window to plausibly
                                 occur" (documented below, not a hidden
                                 assumption).
"""

import argparse
import sys
from datetime import datetime

from parse import parse_tasks, ParsedTask
from pipeline import analyze_task
from scheduler import generate_schedule, TaskForScheduling, PEAK_WINDOWS


def get_multiline_input() -> str:
    print("Paste your task list (syllabus text, bullet list, etc).")
    print("Press Ctrl-D (Linux/Mac) or Ctrl-Z then Enter (Windows) when done:\n")
    lines = sys.stdin.readlines()
    return "".join(lines)


def estimate_in_peak_window(task_deadline: datetime, now: datetime, peak_window: str) -> bool:
    """
    At the point Stage 1+2 needs this flag, no specific time slot has been
    assigned to the task yet (that's the scheduler's job, which runs
    AFTER prediction). So this is a reasonable proxy, not a precise
    measurement: a task is treated as "in peak window" for PREDICTION
    purposes if there's enough runway before its deadline for the user's
    peak window to occur at least once -- i.e. it's plausible to actually
    DO it during peak hours, as opposed to a task due in the next couple
    hours that's forced into whatever slot is available right now.

    This is a simplification worth knowing about: the scheduler's actual
    placement may or may not land a given task in the peak window, and
    this flag doesn't look ahead to check that -- it's an upstream
    estimate the prediction model uses, independent of the final
    schedule. Documented here rather than silently assumed.
    """
    hours_remaining = (task_deadline - now).total_seconds() / 3600
    return hours_remaining >= 4  # at least a few hours of runway


def run_from_parsed_tasks(
    parsed_tasks: list[ParsedTask],
    user_id: str,
    peak_window: str,
    now: datetime,
):
    """
    Shared Stage 1+2+3 runner -- everything that happens AFTER Stage 0+1
    has already produced a list[ParsedTask], regardless of whether those
    tasks came from pasted text (parse_tasks.parse_tasks) or an image
    (vision_parser.parse_tasks_from_image). Both run() and
    run_from_image() below call this so the prediction/scheduling logic
    and its printed output stay in exactly one place.

    Returns the schedule_result dict (same shape /api/schedule/generate
    returns) in case a caller wants to do something with it beyond
    printing, e.g. a future test harness.
    """
    if peak_window not in PEAK_WINDOWS:
        print(f"ERROR: peak_window must be one of {list(PEAK_WINDOWS)}, got '{peak_window}'")
        sys.exit(1)

    if not parsed_tasks:
        print("No tasks were extracted. Nothing to schedule.")
        return None

    for t in parsed_tasks:
        guess_flag = "" if getattr(t, "deadline_matched", True) else "  [GUESSED DEADLINE]"
        print(f"  [{t.task_id}] {t.title!r} | type={t.task_type} | difficulty={t.base_difficulty} "
              f"| deadline={t.deadline.isoformat()} (raw: {t.deadline_raw_text!r}){guess_flag}")

    print("\n" + "=" * 70)
    print("STAGE 1+2: Effort prediction (CatBoost model + correction factor)")
    print("=" * 70)
    predictions = []
    for t in parsed_tasks:
        deadline_hours_remaining = max((t.deadline - now).total_seconds() / 3600, 0.5)
        in_peak = estimate_in_peak_window(t.deadline, now, peak_window)

        result = analyze_task(
            task_id=t.task_id,
            task_type=t.task_type,
            base_difficulty=t.base_difficulty,
            deadline_hours_remaining=deadline_hours_remaining,
            in_peak_window=in_peak,
            user_id=user_id,
        )
        predictions.append((t, result))

        print(f"  [{t.task_id}] {t.title!r}")
        print(f"       deadline_hours_remaining={deadline_hours_remaining:.1f}  in_peak_window={in_peak}")
        print(f"       population_prediction={result['population_predicted_duration_minutes']}min "
              f"  correction_factor={result['correction_factor_applied']}"
              f"  -> final_prediction={result['predicted_duration_minutes']}min")

    print("\n" + "=" * 70)
    print("STAGE 3: Schedule generation")
    print("=" * 70)
    schedule_input = [
        TaskForScheduling(
            task_id=t.task_id,
            title=t.title,
            task_type=t.task_type,
            difficulty=t.base_difficulty,
            predicted_duration_minutes=result["predicted_duration_minutes"],
            deadline=t.deadline,
        )
        for t, result in predictions
    ]

    schedule_result = generate_schedule(schedule_input, peak_window=peak_window, now=now)

    print("\n--- Task Analysis (priority order) ---")
    for ta in schedule_result["task_analysis"]:
        chunk_info = f" -> chunked into {ta['milestones']}" if ta["chunked"] else ""
        print(f"  rank {ta['priority_rank']}: {ta['title']!r} "
              f"(diff={ta['difficulty']}, {ta['predicted_duration_minutes']}min){chunk_info}")

    print("\n--- Time-Blocked Schedule ---")
    for b in schedule_result["schedule"]:
        label = b["title"] or "(break)"
        milestone = f" [{b['milestone']}]" if b["milestone"] else ""
        start = datetime.fromisoformat(b["start_time"]).strftime("%a %H:%M")
        end = datetime.fromisoformat(b["end_time"]).strftime("%H:%M")
        print(f"  {start} -> {end}  [{b['block_type']:5}]  {label}{milestone}")

    print("\n" + "=" * 70)
    print(f"Done. {len(parsed_tasks)} tasks -> {len(schedule_result['schedule'])} schedule blocks.")
    print("=" * 70)

    return schedule_result


def run(raw_text: str, user_id: str, peak_window: str, now: datetime = None):
    if now is None:
        now = datetime.now()

    print("=" * 70)
    print("STAGE 0+1: Parsing raw text via Groq (gpt-oss-20b)")
    print("=" * 70)
    try:
        parsed_tasks: list[ParsedTask] = parse_tasks(raw_text, now=now)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        print("(Check that GROQ_API_KEY is set in your environment)")
        sys.exit(1)

    run_from_parsed_tasks(parsed_tasks, user_id=user_id, peak_window=peak_window, now=now)


def run_from_image(image_path: str, user_id: str, peak_window: str, now: datetime = None):
    """
    Same as run(), but Stage 0+1 goes through vision_parser.py's
    parse_tasks_from_image() instead of parse_tasks.py's parse_tasks() --
    i.e. this is "screenshot in, full schedule out" in one command,
    exercising the exact same Stage 1+2+3 code path (run_from_parsed_tasks)
    as the text CLI, so a judge can watch an image produce a real
    schedule without needing the Flask app or a frontend running.
    """
    if now is None:
        now = datetime.now()

    # Imported here rather than at module level so a plain text-only run
    # of this script doesn't require vision_parser's extra deps (none
    # beyond what parse_tasks.py already needs, currently, but keeping
    # the import scoped to the code path that actually uses it avoids
    # surprises if that changes).
    from vision_parser import parse_tasks_from_image

    print("=" * 70)
    print(f"STAGE 0+1: Parsing image via Groq vision ({image_path})")
    print("=" * 70)
    try:
        parsed_tasks: list[ParsedTask] = parse_tasks_from_image(image_path, now=now)
    except FileNotFoundError:
        print(f"FAILED: image file not found: {image_path}")
        sys.exit(1)
    except ValueError as e:
        print(f"FAILED (bad image): {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        print("(Check that GROQ_API_KEY is set in your environment)")
        sys.exit(1)

    run_from_parsed_tasks(parsed_tasks, user_id=user_id, peak_window=peak_window, now=now)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowState AI — full pipeline CLI test")
    parser.add_argument("--text", type=str, default=None, help="Raw task text (skips interactive prompt)")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a screenshot/image to parse instead of text "
                              "(mutually exclusive with --text; runs the vision path end-to-end)")
    parser.add_argument("--user", type=str, default="cli_test_user", help="User ID (for correction factor lookup)")
    parser.add_argument("--peak", type=str, default="morning", choices=list(PEAK_WINDOWS), help="User's peak window")
    parser.add_argument("--now", type=str, default=None, help="Override 'now' as ISO datetime, e.g. 2026-10-19T08:00:00 (for reproducible testing)")
    args = parser.parse_args()

    if args.text is not None and args.image is not None:
        print("ERROR: pass --text or --image, not both.")
        sys.exit(1)

    now = datetime.fromisoformat(args.now) if args.now else datetime.now()

    if args.image is not None:
        run_from_image(args.image, user_id=args.user, peak_window=args.peak, now=now)
    else:
        raw_text = args.text if args.text is not None else get_multiline_input()
        if not raw_text.strip():
            print("No input provided. Exiting.")
            sys.exit(0)
        run(raw_text, user_id=args.user, peak_window=args.peak, now=now)
