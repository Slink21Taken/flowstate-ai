"""
FlowState AI — Vision Parser (Stage 0 alt input: image ingestion)
======================================================================
Same job as parse_tasks.py's parse_tasks(), but the source is an image
(a screenshot of a syllabus, a Canvas page, a photo of a whiteboard/
planner, etc.) instead of pasted text. This is the "paste a screenshot,
get your tasks" feature.

Deliberately reuses everything from parse_tasks.py that doesn't care
HOW the task JSON was produced: _validate_and_clean_task(),
resolve_deadline(), ParsedTask, VALID_TASK_TYPES. The only new piece is
the Groq call itself, swapped from a text-only chat completion to a
multimodal one (image content block instead of a plain string).

Requires GROQ_API_KEY set as an environment variable -- same rule as
parse_tasks.py, never hardcode it. parse_tasks.py already calls
load_dotenv() at import time, so GROQ_API_KEY is available here too
once this module imports it.

IMPORTANT: same caveat as parse_tasks.py -- api.groq.com was not
reachable from the sandbox this was written in, so the live vision
call below is NOT verified to execute end-to-end from here. Run this
for real, with GROQ_API_KEY set, on your own machine before the demo,
and sanity-check outputs on an actual messy screenshot (not just a
clean test image) -- vision models are noisier on cluttered UI
screenshots (nav bars, sidebars, etc.) than on clean text, so expect
more misses here than on the text path.
"""

import base64
import json
import mimetypes
from typing import Optional
from PIL import Image, UnidentifiedImageError
import io


from datetime import datetime

from parse import (
    ParsedTask,
    SYSTEM_PROMPT,
    _get_client,
    _validate_and_clean_task,
    resolve_deadline,
    _INJECTION_PHRASES,
)
from groq import Groq

# Vision-capable Groq model. Confirmed current as of this writing via
# Groq's own docs (console.groq.com/docs/vision) -- the earlier
# meta-llama/llama-4-scout slug used here was wrong/outdated for Groq
# specifically (that's a model family Groq doesn't currently serve under
# that name). qwen/qwen3.8-27b is Groq's generally-available vision
# model; the other option listed in their docs
# (meta-models/muse-spark-open) is limited-preview and needs Groq to
# grant access, so it's not a safe default. If this model is deprecated/
# renamed again, swap it here; nothing else in this file needs to change.
#
# Groq caps each request to 3 images for this model, and each image
# costs 2,048 input tokens regardless of size -- single-image calls
# (what this file does) are well under that limit.
VISION_MODEL_NAME = "qwen/qwen3.8-27b"

# Output-token cap for the vision call. Groq's rate limiting is partly
# based on REQUESTED output tokens, not just consumed ones -- a
# response_format={"type":"json_object"} call with no max_tokens can
# default to a large requested ceiling and get rejected by the account's
# output-tokens-per-minute (OTPM) limit before the model even runs, as
# happened in testing (requested 1035 > a 1000 OTPM cap on that tier).
# This task's output (a handful of short task objects) rarely needs
# more than a few hundred tokens, so capping it here both avoids that
# rejection and keeps responses cheap/fast. Raise it if real syllabus
# images regularly produce more than ~15-20 tasks and responses start
# getting truncated (look for parsed_json being valid JSON but the
# "tasks" list looking suspiciously cut off).
VISION_MAX_OUTPUT_TOKENS = 800

# Groq (like most providers) caps request payload size -- large PNGs from
# a phone screenshot can be multiple MB. 10MB is a conservative ceiling
# well under typical API limits; fail fast with a clear error rather than
# letting a huge upload hang or get rejected deep in the SDK.
MAX_IMAGE_BYTES = 10 * 1024 * 1024

_SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

def _validate_image_decodes(image_bytes: bytes) -> None:
    """
    Confirms the bytes are an actually-decodable image BEFORE sending
    them to Groq. Without this, a corrupted-but-magic-byte-valid file
    (e.g. a truncated PNG) passes the mimetype check, gets forwarded to
    the Groq vision call, and the model has to attempt (and fail/retry
    on) garbage pixel data server-side -- producing an 8-30+ second hang
    instead of an instant, clean rejection. Image.verify() is cheap and
    fully local; it catches truncated/corrupted files without needing
    a network call to find out they're broken.

    Raises ValueError on anything that doesn't decode -- caller treats
    this the same as any other bad-input ValueError.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"Uploaded file is not a valid, decodable image: {e}")


def _encode_image(image_path: str) -> str:
    """
    Reads an image file from disk and returns a base64 data URL
    (data:image/<type>;base64,<...>) ready to drop into a Groq vision
    message content block.

    Raises ValueError on anything that would silently produce garbage
    downstream -- missing file, unreadable/unknown type, oversized
    upload -- rather than letting a bad encode reach the API and come
    back as a cryptic 4xx.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported or undetected image type for '{image_path}' "
            f"(got {mime_type!r}). Supported: {sorted(_SUPPORTED_MIME_TYPES)}"
        )

    with open(image_path, "rb") as f:
        raw = f.read()

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image '{image_path}' is {len(raw) / 1024 / 1024:.1f}MB, "
            f"exceeds the {MAX_IMAGE_BYTES / 1024 / 1024:.0f}MB limit. "
            "Downscale/compress before uploading."
        )
    _validate_image_decodes(raw)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _call_groq_vision(image_data_url: str, client: Groq) -> dict:
    """
    One multimodal call to Groq, returns the parsed JSON dict. Raises on
    failure (caller handles retry) -- same contract as parse_tasks.py's
    _call_groq(), just with an image content block instead of a plain
    text user message.

    Reuses SYSTEM_PROMPT from parse_tasks.py as-is: the extraction
    schema (title/task_type/deadline_description/base_difficulty) is
    identical regardless of whether the source was text or an image, so
    there's nothing vision-specific to say there. The user-turn text
    below is the only addition, telling the model it's looking at an
    image this time.
    """
    response = client.chat.completions.create(
        model=VISION_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract every task from this image (syllabus "
                            "page, Canvas screenshot, planner photo, etc.) "
                            "using the schema from your instructions."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,  # same rationale as the text path -- extraction,
                          # not creative generation
        max_tokens=VISION_MAX_OUTPUT_TOKENS,  # see constant's comment --
                                                # avoids tripping the
                                                # account's output-token
                                                # rate limit on a request
                                                # with no explicit cap
    )
    content = response.choices[0].message.content
    return json.loads(content)  # may raise json.JSONDecodeError


def _vision_output_looks_suspicious(raw_tasks: list) -> bool:
    """
    Output-side injection heuristic for the vision path. Unlike
    parse_tasks.py's _looks_like_injection_attempt() (which scans the
    RAW INPUT text before the API call), there's no raw text to
    pre-scan here -- the untrusted content is pixels in an image, not a
    string. So this checks the MODEL'S OUTPUT instead: if any extracted
    task's title or raw deadline text contains an injection-associated
    phrase, that's a sign something in the image tried to instruct the
    model and at least partially leaked through into the response,
    worth flagging for review even though schema validation
    (_validate_and_clean_task) already contains the actual damage it
    could do downstream.
    """
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            continue
        text_fields = " ".join(
            str(raw_task.get(k, "")) for k in ("title", "deadline_description")
        ).lower()
        if any(phrase in text_fields for phrase in _INJECTION_PHRASES):
            return True
    return False


def _build_parsed_tasks(raw_tasks: list, now: datetime, flagged: bool = False) -> list[ParsedTask]:
    """
    Shared tail end of both entry points below: validate/clean each raw
    task dict, resolve its deadline, and build the final ParsedTask list.
    Factored out so parse_tasks_from_image() and
    parse_tasks_from_image_bytes() can't drift out of sync on this logic
    the way they would if each inlined it separately.

    resolve_deadline() returns (datetime, matched) -- matched is False
    when the deadline phrase didn't match a known pattern and the 3-day
    fallback was used. Threaded through to deadline_matched on the
    ParsedTask so image-sourced tasks get the same guessed-deadline
    visibility as text-sourced ones, instead of silently defaulting to
    True regardless of what actually happened.

    flagged is OR'd with the per-batch _vision_output_looks_suspicious()
    result (computed by the caller) into source_flagged_for_review on
    every task in the batch, same semantics as the text path.
    """
    output_flagged = flagged or _vision_output_looks_suspicious(raw_tasks)

    results = []
    for i, raw_task in enumerate(raw_tasks):
        cleaned = _validate_and_clean_task(raw_task, i)
        if cleaned is None:
            continue
        deadline, matched = resolve_deadline(cleaned["deadline_raw_text"], now)
        results.append(
            ParsedTask(
                task_id=f"t{i+1}",
                title=cleaned["title"],
                task_type=cleaned["task_type"],
                deadline=deadline,
                base_difficulty=cleaned["base_difficulty"],
                deadline_raw_text=cleaned["deadline_raw_text"],
                deadline_matched=matched,
                source_flagged_for_review=output_flagged,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def parse_tasks_from_image(
    image_path: str,
    now: Optional[datetime] = None,
    max_retries: int = 1,
) -> list[ParsedTask]:
    """
    Vision equivalent of parse_tasks.py's parse_tasks(). Image path in,
    validated ParsedTask objects out -- same return shape, same
    downstream contract (ready to feed into analyze_task() per task,
    same as the text path).

    Deliberately mirrors parse_tasks()'s structure (encode -> call with
    retry -> validate/clean -> resolve deadlines) so the two stay easy
    to compare and keep in sync if the shared schema ever changes.
    """
    if now is None:
        now = datetime.now()

    image_data_url = _encode_image(image_path)  # raises ValueError on bad input,
                                                   # intentionally NOT caught here --
                                                   # a bad file should fail loud, not
                                                   # retry into a wasted API call

    client = _get_client()

    last_error = None
    parsed_json = None
    for attempt in range(max_retries + 1):
        try:
            parsed_json = _call_groq_vision(image_data_url, client)
            break
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001 -- broad on
            # purpose, same as parse_tasks.py: Groq SDK can raise several
            # exception types and a demo shouldn't crash on any of them
            # without at least one retry first.
            last_error = e
            continue

    if parsed_json is None:
        raise RuntimeError(f"Groq vision call failed after {max_retries + 1} attempts: {last_error}")

    raw_tasks = parsed_json.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raw_tasks = []

    return _build_parsed_tasks(raw_tasks, now)


def parse_tasks_from_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    now: Optional[datetime] = None,
    max_retries: int = 1,
) -> list[ParsedTask]:
    """
    Same as parse_tasks_from_image(), but for when the image arrives as
    raw bytes already in memory -- e.g. a Flask request.files upload or
    a clipboard-paste payload from the frontend -- rather than a path on
    disk. Avoids an unnecessary temp-file round trip for the API-server
    use case.

    mime_type must be passed explicitly here since there's no filename
    to sniff it from (mimetypes.guess_type needs a path/extension).
    """
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported image type {mime_type!r}. "
            f"Supported: {sorted(_SUPPORTED_MIME_TYPES)}"
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(image_bytes) / 1024 / 1024:.1f}MB, exceeds the "
            f"{MAX_IMAGE_BYTES / 1024 / 1024:.0f}MB limit."
        )
    _validate_image_decodes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:{mime_type};base64,{b64}"

    if now is None:
        now = datetime.now()

    client = _get_client()

    last_error = None
    parsed_json = None
    for attempt in range(max_retries + 1):
        try:
            parsed_json = _call_groq_vision(image_data_url, client)
            break
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            last_error = e
            continue

    if parsed_json is None:
        raise RuntimeError(f"Groq vision call failed after {max_retries + 1} attempts: {last_error}")

    raw_tasks = parsed_json.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raw_tasks = []

    return _build_parsed_tasks(raw_tasks, now)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# Same split as parse_tasks.py: local/offline checks run for real below;
# the actual Groq vision call needs GROQ_API_KEY + network access and is
# skipped unless both are available.

if __name__ == "__main__":
    import os
    import sys

    print("=== _encode_image() validation tests (fully local, no API needed) ===")

    # missing file
    try:
        _encode_image("/tmp/does_not_exist.png")
        print("  FAIL: should have raised on missing file")
    except FileNotFoundError:
        print("  OK: missing file raises FileNotFoundError")

    # unsupported extension
    try:
        _encode_image(__file__)  # this .py file itself -- wrong mime type
        print("  FAIL: should have raised on unsupported type")
    except ValueError as e:
        print(f"  OK: unsupported type raises ValueError ({e})")

    print("\n=== LIVE vision API test (only runs if GROQ_API_KEY is set AND an image path is given) ===")
    if os.environ.get("GROQ_API_KEY") and len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        try:
            tasks = parse_tasks_from_image(test_image_path)
            for t in tasks:
                print(f"  {json.dumps(t.to_dict(), indent=2)}")
        except Exception as e:
            print(f"  LIVE CALL FAILED: {e}")
    else:
        print(
            "  Skipped -- set GROQ_API_KEY and run as "
            "`python3 vision_parser.py /path/to/screenshot.png` to test for real."
        )
