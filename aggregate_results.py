#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


TASK_METRICS = [
    "consensus_time",
    "scheduling_time",
    "commit_to_start_time",
    "execution_time",
    "termination_time",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate SC26 experiment results."
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results directory. Default: results",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/processed"),
        help="Output directory. Default: results/processed",
    )

    return parser.parse_args()


def mean(values):
    return statistics.mean(values) if values else None


def variance(values):
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def parse_configuration(path: Path):
    """
    Expected path examples:

      results/baseline/b30/n5/gridsearch_times.csv

      results/consensus/q10/b30/n5/gridsearch_times.csv
    """

    scheduler = None
    queue_size = None
    blocks = None
    agents = None

    for part in path.parts:
        if part == "baseline":
            scheduler = "baseline"

        elif part == "consensus":
            scheduler = "consensus"

        elif re.fullmatch(r"q\d+", part):
            queue_size = int(part[1:])

        elif re.fullmatch(r"b\d+", part):
            blocks = int(part[1:])

        elif re.fullmatch(r"n\d+", part):
            agents = int(part[1:])

    if scheduler is None:
        return None

    if blocks is None or agents is None:
        return None

    return {
        "scheduler": scheduler,
        "queue_size": queue_size,
        "blocks": blocks,
        "agents": agents,
    }


def read_gridsearch_times(path):
    values = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            raw = row.get("gridsearch_execution_time", "").strip()

            if not raw:
                continue

            values.append(float(raw))

    return values


def read_task_metrics(path):
    metrics = {
        metric: []
        for metric in TASK_METRICS
    }

    if not path.is_file():
        return metrics

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            for metric in TASK_METRICS:
                raw = (row.get(metric) or "").strip()

                if not raw:
                    continue

                value_ms = float(raw)

                if value_ms < 0:
                    continue

                metrics[metric].append(
                    value_ms / 1000.0
                )

    return metrics


def collect_results(results_dir):
    configurations = defaultdict(
        lambda: {
            "execution_times": [],
            **{
                metric: []
                for metric in TASK_METRICS
            },
        }
    )

    for timing_file in results_dir.rglob(
        "gridsearch_times.csv"
    ):
        config = parse_configuration(
            timing_file
        )

        if config is None:
            print(
                f"Skipping unrecognized path: "
                f"{timing_file}"
            )
            continue

        key = (
            config["scheduler"],
            config["queue_size"],
            config["blocks"],
            config["agents"],
        )

        execution_times = read_gridsearch_times(
            timing_file
        )

        configurations[key][
            "execution_times"
        ].extend(execution_times)

        swarm_file = (
            timing_file.parent
            / "swarm_report.csv"
        )

        task_metrics = read_task_metrics(
            swarm_file
        )

        for metric in TASK_METRICS:
            configurations[key][metric].extend(
                task_metrics[metric]
            )

    return configurations


def build_summary_rows(configurations):
    rows = []

    for key, values in sorted(
        configurations.items(),
        key=lambda item: (
            item[0][0],
            item[0][1] or 0,
            item[0][2],
            item[0][3],
        ),
    ):
        (
            scheduler,
            queue_size,
            blocks,
            agents,
        ) = key

        execution = values[
            "execution_times"
        ]

        row = {
            "scheduler": scheduler,
            "queue_size": (
                ""
                if queue_size is None
                else queue_size
            ),
            "blocks": blocks,
            "agents": agents,
            "execution_mean_s": (
                mean(execution)
                if execution
                else ""
            ),
            "execution_variance_s2": (
                variance(execution)
                if execution
                else ""
            ),
            "execution_std_s": (
                std(execution)
                if execution
                else ""
            ),
            "execution_samples": len(
                execution
            ),
        }

        for metric in TASK_METRICS:
            metric_values = values[metric]

            row[f"{metric}_mean_s"] = (
                mean(metric_values)
                if metric_values
                else ""
            )

            row[
                f"{metric}_variance_s2"
            ] = (
                variance(metric_values)
                if metric_values
                else ""
            )

            row[f"{metric}_std_s"] = (
                std(metric_values)
                if metric_values
                else ""
            )

            row[
                f"{metric}_samples"
            ] = len(metric_values)

        rows.append(row)

    return rows


def build_efficiency_rows(summary_rows):
    baseline = {}

    for row in summary_rows:
        if (
            row["scheduler"] == "baseline"
            and row["agents"] == 1
            and row["execution_mean_s"] != ""
        ):
            baseline[row["blocks"]] = float(
                row["execution_mean_s"]
            )

    rows = []

    for row in summary_rows:
        blocks = row["blocks"]
        agents = row["agents"]

        if blocks not in baseline:
            continue

        if row["execution_mean_s"] == "":
            continue

        execution_time = float(
            row["execution_mean_s"]
        )

        efficiency = (
            baseline[blocks]
            / (
                execution_time
                * agents
            )
        )

        rows.append(
            {
                "scheduler": row[
                    "scheduler"
                ],
                "queue_size": row[
                    "queue_size"
                ],
                "blocks": blocks,
                "agents": agents,
                "baseline_time_s": baseline[
                    blocks
                ],
                "execution_time_s": execution_time,
                "efficiency": efficiency,
            }
        )

    return rows


def write_csv(path, rows):
    if not rows:
        print(
            f"No rows available for {path}"
        )
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    results_dir = args.results.resolve()
    output_dir = args.output.resolve()

    configurations = collect_results(
        results_dir
    )

    if not configurations:
        raise SystemExit(
            "No experiment results found."
        )

    summary_rows = build_summary_rows(
        configurations
    )

    efficiency_rows = build_efficiency_rows(
        summary_rows
    )

    write_csv(
        output_dir
        / "experiment_summary.csv",
        summary_rows,
    )

    write_csv(
        output_dir
        / "efficiency.csv",
        efficiency_rows,
    )

    print(
        f"Configurations found: "
        f"{len(summary_rows)}"
    )

    print(
        f"Wrote: "
        f"{output_dir / 'experiment_summary.csv'}"
    )

    print(
        f"Wrote: "
        f"{output_dir / 'efficiency.csv'}"
    )


if __name__ == "__main__":
    main()
