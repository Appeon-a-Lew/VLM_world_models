#!/usr/bin/env python3
"""Export M0 training logs to TensorBoard and render the results figures."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


ARMS = {
    "fwd": ("Forward-only", "#4cc2ff"),
    "idp": ("Inverse dynamics", "#f6c96a"),
    "ctx": ("Physical context", "#6ee7a8"),
    "both": ("Both", "#f472b6"),
}
BASELINES = {
    "static": ("Static", "#c2cbd9"),
    "const_velocity": ("Constant velocity", "#75849e"),
}
VARIANTS = ("reverse", "half", "double")


def load_results(source):
    paired = source / "paired"
    comparisons = {
        arm: json.loads((paired / f"{arm}_s1.json").read_text())
        for arm in ("idp", "ctx", "both")
    }
    counterfactual = json.loads(
        (paired / "screen_counterfactual" / "counterfactual.json").read_text()
    )["methods"]
    forces = {
        (r["method"], r["speed"]): r
        for r in json.loads((paired / "force_buckets.json").read_text())["rows"]
    }
    effect_skill = json.loads((paired / "effect_skill.json").read_text())
    prompts = {
        arm: json.loads((paired / f"prompt-{arm}.json").read_text())["modes"]
        for arm in ("fwd", "ctx")
    }
    with (paired / "screen_metrics.csv").open(newline="") as file:
        in_distribution = {
            row["method"]: float(row["target_mean"])
            for row in csv.DictReader(file)
            if row["task"] == "B" and row["split"] == "val_id"
        }
    return comparisons, counterfactual, forces, prompts, in_distribution, effect_skill


def write_scalars(writer, step, timestamp, values):
    summary = Summary(value=[
        Summary.Value(tag=tag, simple_value=value) for tag, value in values.items()
    ])
    writer.add_event(Event(step=step, wall_time=timestamp, summary=summary))


def export_tensorboard(destination, arm, records, final_timestamp, evaluation):
    run_dir = destination / f"{arm}_s1"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = final_timestamp - records[-1]["elapsed_s"]
    with TemporaryDirectory(dir=run_dir) as temporary:
        writer = EventFileWriter(temporary)
        try:
            for rec in records:
                values = {
                    "train/loss": rec["loss"],
                    "train/learning_rate": rec["lr"],
                    "train/elapsed_hours": rec["elapsed_s"] / 3600,
                    "tokens/padded_input": rec["tokens"],
                    "tokens/unpadded_input": rec["tokens_in"],
                    "tokens/supervised": rec["tokens_sup"],
                }
                for task, count in sorted(rec["examples_per_task"].items()):
                    values[f"examples/task_{task}"] = count
                write_scalars(writer, rec["step"], started + rec["elapsed_s"], values)
            write_scalars(writer, records[-1]["step"], final_timestamp, evaluation)
        finally:
            writer.close()
        event_file = next(Path(temporary).glob("events.out.tfevents.*"))
        event_file.replace(run_dir / "events.out.tfevents.m0-export")


def save_figure(fig, destination, name):
    fig.savefig(destination / f"{name}.png", dpi=170)
    plt.close(fig)


def training_figure(destination, training):
    fig, (loss, rate, exposure) = plt.subplots(3, 1, figsize=(8.8, 7.8),
                                              sharex=True, layout="constrained")
    for arm, (label, color) in ARMS.items():
        records = training[arm]
        steps = [rec["step"] for rec in records]
        losses = [rec["loss"] for rec in records]
        smoothed = [mean(losses[max(0, i - 8):i + 1]) for i in range(len(losses))]
        loss.plot(steps, losses, color=color, alpha=0.25, linewidth=0.8)
        loss.plot(steps, smoothed, color=color, label=label, linewidth=1.7)
        exposure.plot(steps, [rec["examples_per_task"]["B"] for rec in records], color=color,
                      linestyle="--" if arm in ("ctx", "both") else "-", linewidth=1.4)
    for records in training.values():
        rate.plot([rec["step"] for rec in records], [rec["lr"] for rec in records],
                  color="#c2cbd9", linewidth=1.3)
    loss.set_ylabel("Training loss")
    loss.legend(ncol=2, frameon=False)
    rate.set_ylabel("Learning rate")
    rate.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    exposure.set(ylabel="Task B examples", xlabel="Optimizer step", xlim=(0, 3750))
    exposure.set_xticks([0, 750, 1500, 2250, 3000, 3750])
    save_figure(fig, destination, "training")


def screen_figure(destination, comparisons):
    fig, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color="#8b98ab", linewidth=1)
    threshold = comparisons["both"]["threshold_m"] * 50
    ax.axvline(threshold, color="#c2cbd9", linestyle="--", linewidth=1,
               label=f"Seed-2 minimum gain ({threshold:.2f} cm)")
    for i, (arm, comparison) in enumerate(comparisons.items()):
        seed = comparison["seeds"][0]
        reduction = 100 * seed["reduction_mean"]
        lo, hi = (100 * x for x in seed["reduction_ci95"])
        ax.errorbar(reduction, i, xerr=[[reduction - lo], [hi - reduction]],
                    color=ARMS[arm][1], fmt="o", capsize=4, linewidth=1.8)
    ax.set_yticks(range(len(comparisons)), [ARMS[arm][0] for arm in comparisons])
    ax.set(xlabel="Reduction in matched-pair effect error vs forward-only (cm)",
           ylim=(len(comparisons) - 0.5, -0.5), xlim=(-4, 4))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.03), frameon=False)
    save_figure(fig, destination, "paired-screen")


def effect_skill_figure(destination, effect_skill):
    fig, ax = plt.subplots(figsize=(8.8, 3.7), layout="constrained")
    labels = ("weak", "medium", "strong")
    groups = effect_skill["groups"]
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color="#8b98ab", linewidth=1, linestyle="--")
    for i, (arm, (label, color)) in enumerate(ARMS.items()):
        xs = [j + (i - 1.5) * .16 for j in range(len(labels))]
        rows = [groups[group]["methods"][arm] for group in labels]
        scores = [row["skill"] for row in rows]
        lo, hi = zip(*(row["skill_ci95"] for row in rows))
        ax.errorbar(xs, scores, yerr=[[s - l for s, l in zip(scores, lo)],
                                     [h - s for s, h in zip(scores, hi)]],
                    fmt="o", color=color, capsize=3, label=label)
    ax.set_xticks(range(len(labels)), [f"{label.capitalize()} effects" for label in labels])
    ax.set_ylabel("Skill vs no-change baseline")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=2, loc="lower right", frameon=False)
    save_figure(fig, destination, "effect-skill")


def forces_figure(destination, forces, *, skill=False):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True, layout="constrained")
    methods = {**ARMS, **BASELINES}
    for ax, bucket in zip(axes, ("weak", "strong")):
        ax.grid(axis="y", visible=False)
        for i, (arm, (label, color)) in enumerate(methods.items()):
            row = forces[arm, bucket]
            value = row["skill"] if skill else 100 * row["endpoint_error_m"]
            ax.barh(i, value, color=color, height=0.55)
            text = f"{value:.0%}" if skill else f"{value:.2f}"
            ax.annotate(text, (value, i), xytext=(-4 if value < 0 else 4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        ha="right" if value < 0 else "left")
        ax.set_yticks(range(len(methods)), [label for label, _ in methods.values()])
        ax.set(title=f"{bucket.capitalize()} pushes",
               xlabel="Skill vs static" if skill else "Mean target endpoint error (cm)",
               ylim=(len(methods) - 0.5, -0.5))
        ax.margins(x=0.21)
        if skill:
            ax.axvline(0, color="#8b98ab", linewidth=1, linestyle="--")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save_figure(fig, destination, "force-skill" if skill else "force-buckets")


def action_figure(destination, counterfactual):
    fig, ax = plt.subplots(figsize=(8.8, 3.3), layout="constrained")
    for i, (arm, (label, color)) in enumerate(ARMS.items()):
        variants = counterfactual[arm]["variants"]
        values = [100 * variants[v]["target"]["predicted_change_valid_families_m"]["mean"]
                  for v in VARIANTS]
        ax.bar([j + 0.15 * (i - 1.5) for j in range(len(VARIANTS))], values,
               width=0.13, color=color, label=label)
    truth = counterfactual["fwd"]["variants"]
    ax.scatter(range(len(VARIANTS)), [100 * truth[v]["target"]["gt_effect_m"]["mean"]
                                     for v in VARIANTS],
               marker="D", color="#e7ecf3", label="Simulator", zorder=3)
    ax.set_xticks(range(len(VARIANTS)), [v.capitalize() for v in VARIANTS])
    ax.set(ylabel="Mean target endpoint change (cm)", ylim=(0, 82))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    save_figure(fig, destination, "action-response")


def prompt_figure(destination, prompts, baseline_m):
    fig, ax = plt.subplots(figsize=(8.8, 3.35), layout="constrained")
    modes = {"text": "Original", "nominal": "Nominal\nvector", "recorded": "Recorded\nvector",
             "context": "Recorded vector\n+ context"}
    for arm, results in prompts.items():
        label, color = ARMS[arm]
        ax.plot(range(len(modes)), [100 * results[m]["effect_error_m"] for m in modes],
                color=color, marker="o", label=label)
    ax.set_xticks(range(len(modes)), list(modes.values()))
    ax.set(ylabel="Matched-pair effect error (cm)", ylim=(29, 37))
    baseline_cm = 100 * baseline_m
    skill = ax.secondary_yaxis("right", functions=(lambda cm: 1 - cm / baseline_cm,
                                                    lambda value: baseline_cm * (1 - value)))
    skill.set_ylabel("Skill vs no-change baseline")
    skill.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    save_figure(fig, destination, "prompt-diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Run directory with *_s1 logs and paired/ reports")
    parser.add_argument("output", type=Path, help="Destination for TensorBoard events and ZIP")
    parser.add_argument("--figures", type=Path, help="Figure destination (default: OUTPUT/figures)")
    args = parser.parse_args()
    figures = args.figures or args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    comparisons, counterfactual, forces, prompts, in_distribution, effect_skill = load_results(args.source)
    training = {}
    tensorboard = args.output / "tensorboard"
    for arm in ARMS:
        log = args.source / f"{arm}_s1" / "train_log.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        steps = [rec["step"] for rec in records]
        if steps != sorted(set(steps)) or not steps or steps[-1] != 3750:
            raise ValueError(f"Expected one completed 3,750-step M0 run in {log}")
        training[arm] = records
        groups = effect_skill["groups"]
        evaluation = {
            "final_eval/matched_effect_error_cm": groups["all"]["methods"][arm]["mean_error_m"] * 100,
            "final_eval/in_distribution_target_error_cm": in_distribution[arm] * 100,
        }
        for bucket in ("weak", "strong"):
            row = forces[arm, bucket]
            evaluation[f"final_eval/{bucket}_push_target_error_cm"] = row["endpoint_error_m"] * 100
            evaluation[f"final_eval/{bucket}_push_skill"] = row["skill"]
        for label, group in groups.items():
            evaluation[f"final_eval/matched_effect_skill_{label}"] = group["methods"][arm]["skill"]
        for variant, row in effect_skill["invariance"][arm].items():
            evaluation[f"final_eval/{variant}_within_tolerance_rate"] = row["within_tolerance_rate"]
        export_tensorboard(tensorboard, arm, records, log.stat().st_mtime, evaluation)
    with ZipFile(args.output / "tensorboard.zip", "w", compression=ZIP_DEFLATED) as archive:
        for arm in ARMS:
            file = tensorboard / f"{arm}_s1" / "events.out.tfevents.m0-export"
            archive.write(file, file.relative_to(args.output))
        archive.writestr("README.txt", "M0 seed-1 TensorBoard logs\n\n"
                         "Extract the ZIP and run: tensorboard --logdir tensorboard\n"
                         "The train, tokens and examples tags come from the original training logs.\n"
                         "final_eval tags have one point per run, placed at the final training step.\n"
                         "Skill is 1 minus model error divided by static error within a group.\n"
                         "Validation loss was not logged during training. Training loss mixes the\n"
                         "tasks selected for each run. Wall time is reconstructed from log mtime\n"
                         "and logged elapsed time. Use train/elapsed_hours for elapsed time.\n")

    with plt.rc_context({
        "figure.facecolor": "#11151c", "axes.facecolor": "#11151c",
        "axes.edgecolor": "#333c4c", "axes.labelcolor": "#e7ecf3",
        "text.color": "#e7ecf3", "xtick.color": "#c2cbd9", "ytick.color": "#c2cbd9",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": "#262d3a", "font.size": 10,
    }):
        training_figure(figures, training)
        screen_figure(figures, comparisons)
        effect_skill_figure(figures, effect_skill)
        forces_figure(figures, forces)
        forces_figure(figures, forces, skill=True)
        action_figure(figures, counterfactual)
        prompt_figure(figures, prompts, effect_skill["groups"]["all"]["mean_static_error_m"])


if __name__ == "__main__":
    main()
