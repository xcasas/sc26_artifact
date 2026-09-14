#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


BLOCKS = [30, 50, 100]
QUEUES = [10, 50, 100, 200]


# ============================================================
# Helpers
# ============================================================

def to_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def to_int(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def load_csv(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def save_figure(fig, output_dir, name):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    for extension in ["png", "pdf", "svg"]:
        path = output_dir / f"{name}.{extension}"

        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )

        print(f"Wrote: {path}")

    plt.close(fig)


def select_rows(
    rows,
    scheduler=None,
    queue=None,
    blocks=None,
):
    selected = []

    for row in rows:

        if (
            scheduler is not None
            and row.get("scheduler") != scheduler
        ):
            continue

        if (
            queue is not None
            and to_int(row.get("queue_size")) != queue
        ):
            continue

        if (
            blocks is not None
            and to_int(row.get("blocks")) != blocks
        ):
            continue

        selected.append(row)

    return selected


def get_xy_error(
    rows,
    y_field,
    error_field=None,
):
    rows = sorted(
        rows,
        key=lambda row: (
            to_int(row.get("agents")) or 0
        ),
    )

    x = []
    y = []
    error = []

    for row in rows:
        agents = to_int(
            row.get("agents")
        )

        value = to_float(
            row.get(y_field)
        )

        if agents is None or value is None:
            continue

        x.append(agents)
        y.append(value)

        if error_field is not None:
            error.append(
                to_float(
                    row.get(error_field)
                ) or 0.0
            )

    return x, y, error


# ============================================================
# Baseline vs consensus execution time
# ============================================================

def plot_baseline_vs_consensus(
    summary,
    output_dir,
):
    baseline = select_rows(
        summary,
        scheduler="baseline",
    )

    consensus = select_rows(
        summary,
        scheduler="consensus",
        queue=10,
    )

    if not baseline or not consensus:
        print(
            "Skipping baseline-vs-consensus plots: "
            "both baseline and consensus q10 are required."
        )
        return

    # --------------------------------------------------------
    # One combined figure containing the three workloads
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    plotted = False

    for blocks in BLOCKS:

        baseline_rows = select_rows(
            baseline,
            blocks=blocks,
        )

        consensus_rows = select_rows(
            consensus,
            blocks=blocks,
        )

        x, y, err = get_xy_error(
            baseline_rows,
            "execution_mean_s",
            "execution_std_s",
        )

        if x:
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                linestyle="--",
                label=f"Baseline, {blocks} blocks",
            )

            plotted = True

        x, y, err = get_xy_error(
            consensus_rows,
            "execution_mean_s",
            "execution_std_s",
        )

        if x:
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                label=f"Consensus, {blocks} blocks",
            )

            plotted = True

    if plotted:
        ax.set_xlabel(
            "Number of agents"
        )

        ax.set_ylabel(
            "Execution time (s)"
        )

        ax.set_title(
            "Default scheduler vs. consensus scheduler"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        save_figure(
            fig,
            output_dir,
            "runtime_baseline_vs_consensus",
        )

    else:
        plt.close(fig)

    # --------------------------------------------------------
    # Separate figure for each block size
    # --------------------------------------------------------

    for blocks in BLOCKS:

        baseline_rows = select_rows(
            baseline,
            blocks=blocks,
        )

        consensus_rows = select_rows(
            consensus,
            blocks=blocks,
        )

        if not baseline_rows and not consensus_rows:
            continue

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        plotted = False

        x, y, err = get_xy_error(
            baseline_rows,
            "execution_mean_s",
            "execution_std_s",
        )

        if x:
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                linestyle="--",
                label="Default scheduler",
            )

            plotted = True

        x, y, err = get_xy_error(
            consensus_rows,
            "execution_mean_s",
            "execution_std_s",
        )

        if x:
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                label="Consensus scheduler",
            )

            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(
            "Number of agents"
        )

        ax.set_ylabel(
            "Execution time (s)"
        )

        ax.set_title(
            f"Execution time — {blocks} blocks"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        save_figure(
            fig,
            output_dir,
            f"runtime_baseline_vs_consensus_b{blocks}",
        )


# ============================================================
# Consensus execution time for queue sizes
# ============================================================

def plot_runtime_by_queue(
    summary,
    output_dir,
):
    consensus = select_rows(
        summary,
        scheduler="consensus",
    )

    if not consensus:
        print(
            "Skipping queue-size runtime plots: "
            "no consensus results."
        )
        return

    for blocks in BLOCKS:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        plotted = False

        for queue in QUEUES:

            rows = select_rows(
                consensus,
                queue=queue,
                blocks=blocks,
            )

            x, y, err = get_xy_error(
                rows,
                "execution_mean_s",
                "execution_std_s",
            )

            if not x:
                continue

            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                label=f"Queue {queue}",
            )

            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(
            "Number of agents"
        )

        ax.set_ylabel(
            "Execution time (s)"
        )

        ax.set_title(
            f"Execution time — {blocks} blocks"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        save_figure(
            fig,
            output_dir,
            f"runtime_queues_b{blocks}",
        )


# ============================================================
# Generic consensus/scheduling metric
# ============================================================

def plot_scheduler_metric(
    summary,
    output_dir,
    mean_field,
    std_field,
    ylabel,
    filename,
):
    consensus = select_rows(
        summary,
        scheduler="consensus",
    )

    if not consensus:
        return

    for blocks in BLOCKS:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        plotted = False

        for queue in QUEUES:

            rows = select_rows(
                consensus,
                queue=queue,
                blocks=blocks,
            )

            x, y, err = get_xy_error(
                rows,
                mean_field,
                std_field,
            )

            if not x:
                continue

            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                capsize=3,
                label=f"Queue {queue}",
            )

            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(
            "Number of agents"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{ylabel} — {blocks} blocks"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        save_figure(
            fig,
            output_dir,
            f"{filename}_b{blocks}",
        )


# ============================================================
# Efficiency
# ============================================================

def plot_efficiency(
    efficiency,
    output_dir,
):
    if not efficiency:
        print(
            "Skipping efficiency plots: "
            "no efficiency values available."
        )
        return

    for blocks in BLOCKS:

        rows = [
            row
            for row in efficiency
            if to_int(row.get("blocks")) == blocks
        ]

        if not rows:
            continue

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        plotted = False

        baseline = [
            row
            for row in rows
            if row.get("scheduler") == "baseline"
        ]

        x, y, _ = get_xy_error(
            baseline,
            "efficiency",
        )

        if x:
            ax.plot(
                x,
                y,
                marker="o",
                linestyle="--",
                label="Default scheduler",
            )

            plotted = True

        for queue in QUEUES:

            consensus = [
                row
                for row in rows
                if (
                    row.get("scheduler") == "consensus"
                    and to_int(
                        row.get("queue_size")
                    ) == queue
                )
            ]

            x, y, _ = get_xy_error(
                consensus,
                "efficiency",
            )

            if not x:
                continue

            ax.plot(
                x,
                y,
                marker="o",
                label=f"Consensus q{queue}",
            )

            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.axhline(
            y=1.0,
            linestyle=":",
        )

        ax.set_xlabel(
            "Number of agents"
        )

        ax.set_ylabel(
            "Parallel efficiency"
        )

        ax.set_title(
            f"Parallel efficiency — {blocks} blocks"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        save_figure(
            fig,
            output_dir,
            f"efficiency_b{blocks}",
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate cross-configuration "
            "figures for the SC26 artifact."
        )
    )

    parser.add_argument(
        "--input",
        default="results/processed",
    )

    parser.add_argument(
        "--output",
        default="results/figures",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    summary = load_csv(
        input_dir / "experiment_summary.csv"
    )

    efficiency = load_csv(
        input_dir / "efficiency.csv"
    )

    if not summary:
        print(
            "No experiment summary available."
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Default scheduler vs consensus q10
    plot_baseline_vs_consensus(
        summary,
        output_dir,
    )

    # Effect of admission queue size
    plot_runtime_by_queue(
        summary,
        output_dir,
    )

    # Consensus latency
    plot_scheduler_metric(
        summary,
        output_dir,
        mean_field="consensus_time_mean_s",
        std_field="consensus_time_std_s",
        ylabel="Consensus time (s)",
        filename="consensus_time",
    )

    # Scheduling latency
    plot_scheduler_metric(
        summary,
        output_dir,
        mean_field="scheduling_time_mean_s",
        std_field="scheduling_time_std_s",
        ylabel="Scheduling time (s)",
        filename="scheduling_time",
    )

    # Parallel efficiency
    plot_efficiency(
        efficiency,
        output_dir,
    )

    print()
    print(
        "Cross-configuration figure generation complete."
    )


if __name__ == "__main__":
    main()

