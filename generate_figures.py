#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BLOCKS = [30, 50, 100]
QUEUES = [10, 50, 100, 200]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SC26 paper figures."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/processed"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures"),
    )

    return parser.parse_args()


def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def number(row, key):
    value = row.get(key, "")

    if value in ("", None):
        return None

    return float(value)


def agents_in(rows):
    return sorted(
        {
            int(row["agents"])
            for row in rows
        }
    )


def save(fig, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        path.with_suffix(".png"),
        dpi=300,
    )

    fig.savefig(
        path.with_suffix(".pdf"),
    )

    plt.close(fig)


def plot_default_vs_consensus(
    rows,
    outdir,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for blocks in BLOCKS:
        baseline = [
            row
            for row in rows
            if (
                row["scheduler"]
                == "baseline"
                and int(row["blocks"])
                == blocks
                and int(row["agents"]) > 1
            )
        ]

        consensus = [
            row
            for row in rows
            if (
                row["scheduler"]
                == "consensus"
                and row["queue_size"]
                == "10"
                and int(row["blocks"])
                == blocks
            )
        ]

        baseline.sort(
            key=lambda r: int(r["agents"])
        )

        consensus.sort(
            key=lambda r: int(r["agents"])
        )

        if baseline:
            ax.errorbar(
                [
                    int(r["agents"])
                    for r in baseline
                ],
                [
                    number(
                        r,
                        "execution_mean_s",
                    )
                    for r in baseline
                ],
                yerr=[
                    number(
                        r,
                        "execution_std_s",
                    )
                    or 0
                    for r in baseline
                ],
                marker="o",
                linestyle="--",
                capsize=3,
                label=(
                    f"Baseline, "
                    f"{blocks} blocks"
                ),
            )

        if consensus:
            ax.errorbar(
                [
                    int(r["agents"])
                    for r in consensus
                ],
                [
                    number(
                        r,
                        "execution_mean_s",
                    )
                    for r in consensus
                ],
                yerr=[
                    number(
                        r,
                        "execution_std_s",
                    )
                    or 0
                    for r in consensus
                ],
                marker="o",
                linestyle="-",
                capsize=3,
                label=(
                    f"Consensus, "
                    f"{blocks} blocks"
                ),
            )

    ax.set_xlabel(
        "Number of agents"
    )
    ax.set_ylabel(
        "Execution time (s)"
    )

    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()

    save(
        fig,
        outdir
        / "execution_baseline_vs_consensus",
    )


def plot_queue_sizes(
    rows,
    blocks,
    outdir,
):
    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    for queue in QUEUES:
        selected = [
            row
            for row in rows
            if (
                row["scheduler"]
                == "consensus"
                and row["queue_size"]
                == str(queue)
                and int(row["blocks"])
                == blocks
            )
        ]

        selected.sort(
            key=lambda r: int(r["agents"])
        )

        if not selected:
            continue

        ax.errorbar(
            [
                int(r["agents"])
                for r in selected
            ],
            [
                number(
                    r,
                    "execution_mean_s",
                )
                for r in selected
            ],
            yerr=[
                number(
                    r,
                    "execution_std_s",
                )
                or 0
                for r in selected
            ],
            marker="o",
            capsize=3,
            label=f"Queue {queue}",
        )

    ax.set_xlabel(
        "Number of agents"
    )
    ax.set_ylabel(
        "Execution time (s)"
    )

    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(
        title="Queue size"
    )

    save(
        fig,
        outdir
        / f"queue_sizes_{blocks}_blocks",
    )


def plot_task_metric(
    rows,
    metric,
    ylabel,
    blocks,
    outdir,
):
    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    for queue in QUEUES:
        selected = [
            row
            for row in rows
            if (
                row["scheduler"]
                == "consensus"
                and row["queue_size"]
                == str(queue)
                and int(row["blocks"])
                == blocks
            )
        ]

        selected.sort(
            key=lambda r: int(r["agents"])
        )

        if not selected:
            continue

        means = [
            number(
                r,
                f"{metric}_mean_s",
            )
            for r in selected
        ]

        stds = [
            number(
                r,
                f"{metric}_std_s",
            )
            or 0
            for r in selected
        ]

        valid = [
            (
                int(row["agents"]),
                value,
                deviation,
            )
            for row, value, deviation
            in zip(
                selected,
                means,
                stds,
            )
            if value is not None
        ]

        if not valid:
            continue

        ax.errorbar(
            [v[0] for v in valid],
            [v[1] for v in valid],
            yerr=[v[2] for v in valid],
            marker="o",
            capsize=3,
            label=str(queue),
        )

    ax.set_xlabel(
        "Number of agents"
    )

    ax.set_ylabel(ylabel)

    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    ax.legend(
        title="Queue size"
    )

    save(
        fig,
        outdir
        / f"{metric}_{blocks}_blocks",
    )


def plot_efficiency(
    rows,
    outdir,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for scheduler in (
        "baseline",
        "consensus",
    ):
        for blocks in BLOCKS:
            selected = [
                row
                for row in rows
                if (
                    row["scheduler"]
                    == scheduler
                    and int(row["blocks"])
                    == blocks
                    and (
                        scheduler
                        == "baseline"
                        or row[
                            "queue_size"
                        ]
                        == "10"
                    )
                )
            ]

            selected.sort(
                key=lambda r: int(
                    r["agents"]
                )
            )

            if not selected:
                continue

            linestyle = (
                "--"
                if scheduler
                == "baseline"
                else "-"
            )

            ax.plot(
                [
                    int(r["agents"])
                    for r in selected
                ],
                [
                    float(
                        r["efficiency"]
                    )
                    for r in selected
                ],
                marker="o",
                linestyle=linestyle,
                label=(
                    f"{scheduler.capitalize()}, "
                    f"{blocks} blocks"
                ),
            )

    ax.set_xlabel(
        "Number of agents"
    )

    ax.set_ylabel(
        "Parallel efficiency"
    )

    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()

    save(
        fig,
        outdir / "parallel_efficiency",
    )


def main():
    args = parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    summary = load_csv(
        input_dir
        / "experiment_summary.csv"
    )

    efficiency = load_csv(
        input_dir
        / "efficiency.csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_default_vs_consensus(
        summary,
        output_dir,
    )

    for blocks in BLOCKS:
        plot_queue_sizes(
            summary,
            blocks,
            output_dir,
        )

        plot_task_metric(
            summary,
            "consensus_time",
            "Consensus time (s)",
            blocks,
            output_dir,
        )

        plot_task_metric(
            summary,
            "scheduling_time",
            "Scheduling time (s)",
            blocks,
            output_dir,
        )

    plot_efficiency(
        efficiency,
        output_dir,
    )

    print(
        f"Figures written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
