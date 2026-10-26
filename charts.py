"""
FlowState AI — Competition Chart Suite
===========================================
Generates 8 matplotlib charts for the pitch/demo. Each function is
self-contained and documents exactly what real data to feed it (pulled
from your actual catboost_model.cbm, correction_factors.json,
train_model.py's data generator, generate_feasibility_data.py, and
synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv).

Where real data isn't trivially available (e.g. you'd need to re-run a
generator), each function accepts plain arrays/DataFrames so you can glue
it to whatever your actual pipeline produces -- nothing here fabricates
numbers for you. Swap the "# === REPLACE WITH REAL DATA ===" blocks in
__main__ with your real loading code.

Run:
    pip install matplotlib numpy pandas --break-system-packages
    python3 make_charts.py

Outputs land in ./charts/*.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.family"] = "sans-serif"

OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

TASK_TYPES = ["coding", "essay", "reading", "math", "creative"]
TASK_TYPE_COLORS = {
    "coding": "#2563eb", "essay": "#dc2626", "reading": "#16a34a",
    "math": "#9333ea", "creative": "#ea580c",
}
ARCHETYPE_COLORS = {
    "reliable": "#16a34a", "chronically_late": "#dc2626",
    "crunch_sensitive": "#ea580c", "erratic": "#6b7280",
}


# ---------------------------------------------------------------------------
# 1. Predicted vs. Actual duration (CatBoost validity)
# ---------------------------------------------------------------------------

def chart_predicted_vs_actual(y_true, y_pred, out=f"{OUT_DIR}/01_pred_vs_actual.png"):
    """
    y_true, y_pred: array-likes of actual vs. predicted duration_minutes
    on a HELD-OUT split (not training data -- pull this from
    train_model.py's own validation set, or re-predict with
    EffortPredictor.predict_population_duration() against your
    synthetic generator's known ground-truth durations).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=14, color="#2563eb", edgecolors="none")

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#ef4444",
             linewidth=1.5, label="Perfect prediction")

    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    mae = np.mean(np.abs(y_true - y_pred))
    ax.text(0.04, 0.96, f"R\u00b2 = {r2:.3f}\nMAE = {mae:.1f} min\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))

    ax.set_xlabel("Actual duration (min)")
    ax.set_ylabel("Predicted duration (min)")
    ax.set_title("CatBoost: Predicted vs. Actual Task Duration")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 2. Feature importance (CatBoost)
# ---------------------------------------------------------------------------

def chart_feature_importance(feature_names, importances, out=f"{OUT_DIR}/02_feature_importance.png"):
    """
    feature_names: list of str, e.g. pipeline.py's FEATURE_COLS
    importances: list/array of floats, same order -- from
    model.get_feature_importance() on your loaded CatBoostRegressor
    (EffortPredictor.model.get_feature_importance()).
    """
    order = np.argsort(importances)
    names = np.array(feature_names)[order]
    vals = np.array(importances)[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2563eb" if "deadline" in n or "crunch" in n or "far_out" in n else "#94a3b8" for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("Feature importance")
    ax.set_title("What Drives the Duration Prediction")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 3. Feasibility classifier: threshold tradeoff
# ---------------------------------------------------------------------------

def chart_feasibility_threshold_tradeoff(
    thresholds, precision_vals, recall_vals, f1_vals, chosen_threshold,
    out=f"{OUT_DIR}/03_feasibility_threshold.png",
):
    """
    thresholds: array of candidate decision thresholds (e.g. np.linspace(0,1,101))
    precision_vals, recall_vals, f1_vals: same-length arrays, computed for
    the INFEASIBLE class at each threshold on your validation split
    (sklearn.metrics.precision_recall_fscore_support with pos_label=0 or
    whichever label means "infeasible", swept over thresholds).
    chosen_threshold: the actual value saved in feasibility_threshold.joblib
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precision_vals, label="Precision (infeasible)", color="#2563eb")
    ax.plot(thresholds, recall_vals, label="Recall (infeasible)", color="#dc2626")
    ax.plot(thresholds, f1_vals, label="F1 (infeasible)", color="#16a34a", linestyle="--")
    ax.axvline(chosen_threshold, color="#111827", linestyle=":", linewidth=1.5)
    ax.text(chosen_threshold + 0.01, 0.05, f"chosen\nthreshold={chosen_threshold:.2f}",
            fontsize=9, color="#111827")
    ax.axvline(0.5, color="#9ca3af", linestyle=":", linewidth=1)
    ax.text(0.5 + 0.01, 0.85, "default 0.5", fontsize=8, color="#9ca3af")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Feasibility Classifier: We Optimized for Catching Risk, Not Accuracy")
    ax.legend(loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def chart_feasibility_confusion_matrix(
    cm_default, cm_chosen, out=f"{OUT_DIR}/03b_feasibility_confusion.png",
):
    """
    cm_default, cm_chosen: 2x2 arrays [[TN, FP], [FN, TP]] (sklearn.metrics.confusion_matrix
    output) at threshold=0.5 vs. your chosen threshold, on the same validation split.
    Rows = actual (on-time, infeasible), Cols = predicted (on-time, infeasible).
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ["Default threshold (0.5)", "Chosen threshold (balanced)"]
    for ax, cm, title in zip(axes, [cm_default, cm_chosen], titles):
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["On-time", "Infeasible"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["On-time", "Infeasible"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Catching Infeasible Schedules: Default vs. Chosen Threshold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 4. Correction factor convergence (personalization story)
# ---------------------------------------------------------------------------

def chart_correction_factor_convergence(
    sample_indices, reliable_factors, chronically_late_factors,
    out=f"{OUT_DIR}/04_factor_convergence.png",
):
    """
    sample_indices: e.g. range(1, 11) -- completion number
    reliable_factors / chronically_late_factors: the effective_factor()
    value after each completion, for two synthetic users you drive through
    /api/tasks/complete with a consistent ratio (e.g. ~1.0 for reliable,
    ~1.5 for chronically late). This is literally what
    test_correction_factor_converges_near_2x_for_consistent_2x_ratio
    computes -- just log effective_factor at each step instead of only
    asserting the final value.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sample_indices, reliable_factors, marker="o", color="#16a34a",
            label="Reliable student (ratio \u2248 1.0)")
    ax.plot(sample_indices, chronically_late_factors, marker="o", color="#dc2626",
            label="Chronically-late student (ratio \u2248 1.5)")
    ax.axhline(1.0, color="#9ca3af", linestyle=":", linewidth=1)
    ax.set_xlabel("Completed tasks for this user/task_type")
    ax.set_ylabel("Effective correction factor")
    ax.set_title("The Model Personalizes Over Time")
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 5. "Does it get smarter" -- feasibility probability before/after history
# ---------------------------------------------------------------------------

def chart_feasibility_before_after(
    p_before, p_after, out=f"{OUT_DIR}/05_feasibility_before_after.png",
):
    """
    p_before, p_after: the two completion_probability floats from
    demo_feasibility.py's cold-start-vs-with-history scenario (same
    schedule request, before vs. after 5 on-time completions). Matches
    test_feasibility_improves_or_changes_after_real_completion_history.
    """
    fig, ax = plt.subplots(figsize=(4.5, 5))
    bars = ax.bar(["Before\n(cold start)", "After\n(5 completions)"], [p_before, p_after],
                   color=["#9ca3af", "#16a34a"], width=0.55)
    for b, v in zip(bars, [p_before, p_after]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    delta = p_after - p_before
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Completion probability")
    ax.set_title(f"Same Schedule, Same Deadline \u2014\n\u0394 = {delta:+.2f} from History Alone")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 6. Pipeline diagram (Stage 0 -> 6)
# ---------------------------------------------------------------------------

def chart_pipeline_diagram(out=f"{OUT_DIR}/06_pipeline_diagram.png"):
    """No real data needed -- static architecture diagram built from the
    actual stage breakdown in app.py / TODO_backend_ml.md."""
    stages = [
        ("Stage 0+1\nIngestion", "Groq LLM\n(gpt-oss-20b)", "#f59e0b"),
        ("Stage 1+2\nDuration", "CatBoost +\ncorrection factor", "#2563eb"),
        ("Stage 3\nSchedule", "Rule-based\nscheduler", "#16a34a"),
        ("Stage 4\nLearning", "EMA correction\nupdate", "#9333ea"),
        ("Stage 5\nFeasibility", "Logistic\nRegression", "#dc2626"),
        ("Stage 6\nPersistence", "JSON-backed\npresets", "#6b7280"),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    box_w, box_h, gap = 1.9, 1.3, 0.35
    x = 0
    for i, (title, sub, color) in enumerate(stages):
        rect = mpatches.FancyBboxPatch((x, 0), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        linewidth=1.5, edgecolor=color, facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(x + box_w / 2, box_h * 0.68, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + box_w / 2, box_h * 0.3, sub, ha="center", va="center", fontsize=8.5, color="#374151")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap, box_h / 2), xytext=(x + box_w, box_h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"))
        x += box_w + gap
    ax.set_xlim(-0.2, x)
    ax.set_ylim(-0.2, box_h + 0.3)
    ax.axis("off")
    ax.set_title("FlowState Pipeline: What's LLM, What's Trained Model, What's Rules", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 7. Example generated schedule (Gantt-style timeline)
# ---------------------------------------------------------------------------

def chart_schedule_gantt(schedule_blocks, out=f"{OUT_DIR}/07_schedule_gantt.png"):
    """
    schedule_blocks: list of dicts matching the real /api/schedule/generate
    response's "schedule" list -- each needs start_time, end_time (ISO
    strings), block_type ("work"/"break"/"buffer"), title (or None).
    Just pass r.json()["schedule"] from a real call straight in.

    Rendered in chronological order (earliest block at top) with a
    HH:MM-level tick formatter and a one-line subtitle explaining what
    the colors mean, so the chart is readable at a glance without a
    presenter narrating it.
    """
    df = pd.DataFrame(schedule_blocks)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60

    # Chronological order top-to-bottom reads like a real day planner --
    # the API's own list order isn't guaranteed to be time-sorted.
    df = df.sort_values("start_time").reset_index(drop=True)

    color_map = {"work": "#2563eb", "break": "#d1d5db", "buffer": "#fca5a5"}
    labels = []
    for _, row in df.iterrows():
        title = row.get("title")
        # pandas turns JSON null into NaN (a float), not None/empty-string,
        # so a plain `or` fallback doesn't catch it -- check explicitly.
        if pd.isna(title) or not title:
            lbl = str(row["block_type"]).capitalize()
        else:
            lbl = str(title)
        milestone = row.get("milestone")
        if milestone is not None and not pd.isna(milestone):
            lbl += f" ({milestone})"
        labels.append(lbl)
    df["label"] = labels

    # Keep each row's y-position in chronological order even though
    # matplotlib's barh would otherwise plot bottom-to-top -- invert the
    # axis afterward so block 1 is at the TOP, like a day planner.
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.4 * len(df))))
    for i, row in df.iterrows():
        ax.barh(row["label"], row["duration"],
                left=mdates.date2num(row["start_time"]),
                color=color_map.get(row["block_type"], "#9ca3af"), edgecolor="white", height=0.7)
    ax.invert_yaxis()

    ax.xaxis_date()
    # Force a tick roughly every 2 hours so labels show real time-of-day
    # instead of all collapsing onto midnight with daily-spaced auto-ticks.
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %I:%M %p"))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.set_xlabel("Time")
    ax.set_title("Example Generated Schedule", fontsize=13, fontweight="bold", pad=18)
    ax.text(0.5, 1.02,
            "One task \u2192 broken into chunks, work blocks, and adaptive breaks \u2014 placed around the deadline",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#4b5563")

    handles = [mpatches.Patch(color=c, label=k.capitalize()) for k, c in color_map.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 8. Performance scaling (100 vs 1000 tasks)
# ---------------------------------------------------------------------------

def chart_performance_scaling(task_counts, elapsed_seconds, out=f"{OUT_DIR}/08_performance_scaling.png"):
    """
    task_counts: e.g. [1, 10, 50, 100, 500, 1000]
    elapsed_seconds: measured wall-clock time for each, from timing
    /api/schedule/generate calls directly (your test suite already times
    the 100 and 1000 cases -- add a couple more points for a real curve).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(task_counts, elapsed_seconds, marker="o", color="#2563eb", label="Measured")
    # naive linear reference from the first real datapoint
    if len(task_counts) >= 2 and task_counts[0] > 0:
        slope = elapsed_seconds[0] / task_counts[0] if task_counts[0] else 0
        ref = [slope * n for n in task_counts]
        ax.plot(task_counts, ref, linestyle="--", color="#9ca3af", label="Linear reference")
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Response time (s)")
    ax.set_title("Scheduler Scales Linearly, Not Quadratically")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 9. Synthetic archetype distributions (training data explainer)
# ---------------------------------------------------------------------------

def chart_archetype_distributions(df, out=f"{OUT_DIR}/09_archetype_distributions.png"):
    """
    df: pandas DataFrame loaded from
    synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv (or
    equivalent), needs columns: 'archetype' and 'actual_over_predicted_ratio'
    (or whatever your generator calls the actual/predicted ratio column --
    rename to match). If your CSV doesn't have this ratio precomputed,
    derive it from actual_duration_minutes / predicted_duration_minutes
    columns first.
    """
    archetypes = df["archetype"].unique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arch in archetypes:
        subset = df[df["archetype"] == arch]["actual_over_predicted_ratio"]
        ax.hist(subset, bins=30, alpha=0.5, label=arch.replace("_", " ").title(),
                color=ARCHETYPE_COLORS.get(arch, "#6b7280"), density=True)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Actual / Predicted duration ratio")
    ax.set_ylabel("Density")
    ax.set_title("Synthetic Student Archetypes: Behavior Distributions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 10. Population duration formula (ground-truth generating function)
# ---------------------------------------------------------------------------

def chart_population_duration_formula(
    base_duration_at_diff5, difficulty_scale_exponent,
    out=f"{OUT_DIR}/10_population_formula.png",
):
    """
    base_duration_at_diff5: dict {task_type: minutes_at_difficulty_5},
    pulled straight from train_model.py's own constants.
    difficulty_scale_exponent: the exponent constant from the same file.

    Formula assumed: duration(d) = base_duration_at_diff5 * (d / 5) ** exponent
    -- ADJUST this if train_model.py's actual closed form differs; this
    is a best-effort reconstruction from the description in
    TODO_backend_ml.md ("BASE_DURATION_AT_DIFF5 x DIFFICULTY_SCALE_EXPONENT").
    """
    difficulties = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tt, base in base_duration_at_diff5.items():
        durations = base * (difficulties / 5.0) ** difficulty_scale_exponent
        ax.plot(difficulties, durations, marker="o", color=TASK_TYPE_COLORS.get(tt, "#000"), label=tt)
    ax.set_xlabel("Base difficulty (1-10)")
    ax.set_ylabel("Population duration (min)")
    ax.set_title("Ground-Truth Duration Formula by Task Type\n(what the regressor is trained to approximate)")
    ax.set_xticks(difficulties)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 11. Synthetic dataset composition / class balance
# ---------------------------------------------------------------------------

def chart_dataset_composition(df, out=f"{OUT_DIR}/11_dataset_composition.png"):
    """
    df: the full synthetic feasibility dataset (same CSV as #9), needs
    columns 'archetype' and 'completed_on_time' (bool/0-1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    arch_counts = df["archetype"].value_counts()
    axes[0].bar(arch_counts.index.str.replace("_", " ").str.title(), arch_counts.values,
                color=[ARCHETYPE_COLORS.get(a, "#6b7280") for a in arch_counts.index])
    axes[0].set_title("Rows per Archetype")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    class_counts = df["completed_on_time"].value_counts().sort_index()
    labels = ["Infeasible\n(late)", "On-time"]
    colors = ["#dc2626", "#16a34a"]
    pct = 100 * class_counts / class_counts.sum()
    axes[1].bar(labels, class_counts.values, color=colors)
    for i, (v, p) in enumerate(zip(class_counts.values, pct.values)):
        axes[1].text(i, v + max(class_counts.values) * 0.02, f"{p:.0f}%", ha="center", fontsize=11)
    axes[1].set_title("Class Balance (why we used class_weight='balanced')")
    axes[1].set_ylabel("Count")

    fig.suptitle("Synthetic Feasibility Training Data")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 12. Crunch sensitivity by archetype (does synthetic data behave sanely)
# ---------------------------------------------------------------------------

def chart_crunch_sensitivity_by_archetype(df, out=f"{OUT_DIR}/12_crunch_sensitivity.png"):
    """
    df: same CSV, needs 'archetype', 'is_crunch' (bool/0-1), and
    'actual_over_predicted_ratio' columns.
    Shows that crunch-sensitive archetype's spread widens under crunch
    while reliable archetype's doesn't -- sanity check that the synthetic
    generator encodes a real, differentiated behavioral signal rather
    than being noise with labels on it.
    """
    archetypes = df["archetype"].unique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = []
    data = []
    tick_labels = []
    pos = 0
    for arch in archetypes:
        for crunch_val, crunch_label in [(0, "Normal"), (1, "Crunch")]:
            subset = df[(df["archetype"] == arch) & (df["is_crunch"] == crunch_val)]["actual_over_predicted_ratio"]
            if len(subset) > 0:
                data.append(subset.values)
                positions.append(pos)
                tick_labels.append(f"{arch.replace('_',' ').title()}\n{crunch_label}")
                pos += 1
        pos += 0.5

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, lbl in zip(bp["boxes"], tick_labels):
        patch.set_facecolor("#fca5a5" if "Crunch" in lbl else "#93c5fd")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=15)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Actual / Predicted ratio")
    ax.set_title("Crunch Sensitivity Differs by Archetype\n(sanity check on synthetic data realism)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Example usage -- replace with real data loading
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("This script only defines chart functions.")
    print("Import it and call each chart_*() with your real data, e.g.:\n")
    print("""
    import make_charts as mc
    import pandas as pd
    import requests

    # --- 1 & 2: load your real model ---
    from pipeline import EffortPredictor, FEATURE_COLS
    predictor = EffortPredictor()
    # build y_true / y_pred from a held-out split, then:
    # mc.chart_predicted_vs_actual(y_true, y_pred)
    importances = predictor.model.get_feature_importance()
    mc.chart_feature_importance(FEATURE_COLS, importances)

    # --- 3: feasibility threshold sweep ---
    # from your training script, sweep thresholds + sklearn.metrics.precision_recall_fscore_support
    # mc.chart_feasibility_threshold_tradeoff(thresholds, precisions, recalls, f1s, chosen_threshold=0.XX)

    # --- 4: correction factor convergence ---
    # drive two fresh users through /api/tasks/complete N times each, log effective_factor
    # mc.chart_correction_factor_convergence(range(1,11), reliable_list, late_list)

    # --- 5: before/after feasibility ---
    # from demo_feasibility.py's cold-start scenario
    # mc.chart_feasibility_before_after(p_before, p_after)

    # --- 6: pipeline diagram (no data needed) ---
    mc.chart_pipeline_diagram()

    # --- 7: real schedule gantt ---
    r = requests.post("http://localhost:5000/api/schedule/generate", json={...})
    mc.chart_schedule_gantt(r.json()["schedule"])

    # --- 8: performance scaling ---
    # time /api/schedule/generate at several task counts
    # mc.chart_performance_scaling([1,10,50,100,500,1000], [...])

    # --- 9, 11, 12: synthetic dataset charts ---
    df = pd.read_csv("synthetic_feasibility_data_with_archetype_DEBUG_ONLY.csv")
    # df["actual_over_predicted_ratio"] = df["actual_duration_minutes"] / df["predicted_duration_minutes"]
    mc.chart_archetype_distributions(df)
    mc.chart_dataset_composition(df)
    mc.chart_crunch_sensitivity_by_archetype(df)

    # --- 10: ground-truth formula ---
    # from train_model.py's own constants
    # mc.chart_population_duration_formula(BASE_DURATION_AT_DIFF5, DIFFICULTY_SCALE_EXPONENT)
    """)
