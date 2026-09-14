#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_TASKS_PER_POINT = 3000


TIMELINE_COLUMNS = [
    "elapsed_s",
    "interval_start_s",
    "interval_end_s",

    "consensus_samples",
    "consensus_time_mean_ms",
    "consensus_time_variance_ms2",
    "consensus_time_std_ms",
    "consensus_throughput_tasks_per_min",

    "scheduling_samples",
    "scheduling_time_mean_ms",
    "scheduling_time_variance_ms2",
    "scheduling_time_std_ms",

    "max_parallel_consensus",
    "max_parallel_scheduling",
]


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


def read_csv(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TIMELINE_COLUMNS,
        )

        writer.writeheader()

        if rows:
            writer.writerows(rows)

    print(f"Wrote: {path}")


def save_figure(fig, output_dir, name):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    for extension in ["png", "pdf", "svg"]:
        path = (
            output_dir
            / f"{name}.{extension}"
        )

        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )

        print(f"Wrote: {path}")

    plt.close(fig)


# ============================================================
# Path configuration
# ============================================================

def parse_configuration(path):
    scheduler = None
    queue = None
    blocks = None
    agents = None

    for part in path.parts:

        if part == "consensus":
            scheduler = "consensus"

        elif part == "baseline":
            scheduler = "baseline"

        elif re.fullmatch(r"q\d+", part):
            queue = int(part[1:])

        elif re.fullmatch(r"b\d+", part):
            blocks = int(part[1:])

        elif re.fullmatch(r"n\d+", part):
            agents = int(part[1:])

    if scheduler != "consensus":
        return None

    if blocks is None or agents is None:
        return None

    return {
        "scheduler": scheduler,
        "queue": queue,
        "blocks": blocks,
        "agents": agents,
    }


# ============================================================
# Pooled statistics
# ============================================================

def pooled_statistics(
    rows,
    count_field,
    mean_field,
    std_field,
):
    groups = []

    for row in rows:
        n = to_int(
            row.get(count_field)
        )

        mean = to_float(
            row.get(mean_field)
        )

        std = to_float(
            row.get(std_field)
        )

        if n is None or mean is None:
            continue

        if n <= 0:
            continue

        if std is None:
            std = 0.0

        groups.append(
            (
                n,
                mean,
                std,
            )
        )

    if not groups:
        return (
            0,
            None,
            None,
            None,
        )

    total_n = sum(
        n
        for n, _, _ in groups
    )

    pooled_mean = (
        sum(
            n * mean
            for n, mean, _ in groups
        )
        / total_n
    )

    if total_n == 1:
        return (
            total_n,
            pooled_mean,
            0.0,
            0.0,
        )

    ss = 0.0

    for n, mean, std in groups:

        if n > 1:
            ss += (
                (n - 1)
                * std ** 2
            )

        ss += (
            n
            * (
                mean
                - pooled_mean
            ) ** 2
        )

    variance = (
        ss
        / (total_n - 1)
    )

    std = math.sqrt(
        variance
    )

    return (
        total_n,
        pooled_mean,
        variance,
        std,
    )


# ============================================================
# Window timestamps
# ============================================================

def row_start(row):
    candidates = [
        to_int(
            row.get(
                "consensus_window_start_ms"
            )
        ),
        to_int(
            row.get(
                "scheduling_window_start_ms"
            )
        ),
    ]

    candidates = [
        value
        for value in candidates
        if value is not None
    ]

    if not candidates:
        return None

    return min(candidates)


def row_end(row):
    candidates = [
        to_int(
            row.get(
                "consensus_window_end_ms"
            )
        ),
        to_int(
            row.get(
                "scheduling_window_end_ms"
            )
        ),
    ]

    candidates = [
        value
        for value in candidates
        if value is not None
    ]

    if not candidates:
        return None

    return max(candidates)


# ============================================================
# Aggregate timeline group
# ============================================================

def aggregate_group(
    rows,
    origin_ms,
):
    starts = [
        row_start(row)
        for row in rows
    ]

    starts = [
        value
        for value in starts
        if value is not None
    ]

    ends = [
        row_end(row)
        for row in rows
    ]

    ends = [
        value
        for value in ends
        if value is not None
    ]

    if not starts or not ends:
        return None

    start_ms = min(starts)
    end_ms = max(ends)

    (
        consensus_samples,
        consensus_mean,
        consensus_variance,
        consensus_std,
    ) = pooled_statistics(
        rows,
        "consensus_samples",
        "consensus_time_mean_ms",
        "consensus_time_std_ms",
    )

    (
        scheduling_samples,
        scheduling_mean,
        scheduling_variance,
        scheduling_std,
    ) = pooled_statistics(
        rows,
        "scheduling_samples",
        "scheduling_time_mean_ms",
        "scheduling_time_std_ms",
    )

    duration_s = (
        end_ms - start_ms
    ) / 1000.0

    if duration_s > 0:
        throughput = (
            consensus_samples
            / duration_s
            * 60.0
        )
    else:
        throughput = 0.0

    max_parallel_consensus_values = [
        to_int(
            row.get(
                "max_parallel_consensus"
            )
        )
        for row in rows
    ]

    max_parallel_consensus_values = [
        value
        for value
        in max_parallel_consensus_values
        if value is not None
    ]

    max_parallel_scheduling_values = [
        to_int(
            row.get(
                "max_parallel_scheduling"
            )
        )
        for row in rows
    ]

    max_parallel_scheduling_values = [
        value
        for value
        in max_parallel_scheduling_values
        if value is not None
    ]

    elapsed_s = (
        end_ms - origin_ms
    ) / 1000.0

    return {
        "elapsed_s": elapsed_s,

        "interval_start_s": (
            start_ms - origin_ms
        ) / 1000.0,

        "interval_end_s": (
            end_ms - origin_ms
        ) / 1000.0,

        "consensus_samples": (
            consensus_samples
        ),

        "consensus_time_mean_ms": (
            consensus_mean
            if consensus_mean is not None
            else ""
        ),

        "consensus_time_variance_ms2": (
            consensus_variance
            if consensus_variance is not None
            else ""
        ),

        "consensus_time_std_ms": (
            consensus_std
            if consensus_std is not None
            else ""
        ),

        "consensus_throughput_tasks_per_min": (
            throughput
        ),

        "scheduling_samples": (
            scheduling_samples
        ),

        "scheduling_time_mean_ms": (
            scheduling_mean
            if scheduling_mean is not None
            else ""
        ),

        "scheduling_time_variance_ms2": (
            scheduling_variance
            if scheduling_variance is not None
            else ""
        ),

        "scheduling_time_std_ms": (
            scheduling_std
            if scheduling_std is not None
            else ""
        ),

        "max_parallel_consensus": (
            max(
                max_parallel_consensus_values
            )
            if max_parallel_consensus_values
            else ""
        ),

        "max_parallel_scheduling": (
            max(
                max_parallel_scheduling_values
            )
            if max_parallel_scheduling_values
            else ""
        ),
    }


# ============================================================
# Timeline creation
# ============================================================

def build_timeline(rows, tasks_per_point=None):
    usable = [
        row
        for row in rows
        if (
            row_start(row) is not None
            and row_end(row) is not None
        )
    ]

    if not usable:
        return []

    usable.sort(
        key=lambda row: (
            row_end(row),
            row.get("source_host", ""),
        )
    )

    origin_ms = min(
        row_start(row)
        for row in usable
    )

    # Group rows that correspond to the same time window
    # across different agents.
    windows = {}

    for row in usable:
        start = row_start(row)
        end = row_end(row)

        key = (
            start,
            end,
        )

        windows.setdefault(
            key,
            [],
        ).append(row)

    timeline = []

    for (
        start_ms,
        end_ms,
    ), window_rows in sorted(
        windows.items(),
        key=lambda item: item[0][1],
    ):

        point = aggregate_group(
            window_rows,
            origin_ms,
        )

        if point is not None:
            timeline.append(point)

    return timeline

# ============================================================
# Timeline plots
# ============================================================

def plot_consensus_time(
    timeline,
    output_dir,
):
    x = []
    y = []
    error = []

    for row in timeline:

        elapsed = to_float(
            row["elapsed_s"]
        )

        mean = to_float(
            row[
                "consensus_time_mean_ms"
            ]
        )

        std = to_float(
            row[
                "consensus_time_std_ms"
            ]
        )

        if elapsed is None or mean is None:
            continue

        x.append(
            elapsed / 60.0
        )

        y.append(
            mean / 1000.0
        )

        error.append(
            (std or 0.0)
            / 1000.0
        )

    if not x:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.errorbar(
        x,
        y,
        yerr=error,
        marker="o",
        capsize=3,
    )

    ax.set_xlabel(
        "Elapsed time (min)"
    )

    ax.set_ylabel(
        "Consensus time (s)"
    )

    ax.set_title(
        "Consensus time during execution"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "consensus_time_timeline",
    )


def plot_consensus_throughput(
    timeline,
    output_dir,
):
    x = []
    y = []

    for row in timeline:

        elapsed = to_float(
            row["elapsed_s"]
        )

        throughput = to_float(
            row[
                "consensus_throughput_tasks_per_min"
            ]
        )

        if (
            elapsed is None
            or throughput is None
        ):
            continue

        x.append(
            elapsed / 60.0
        )

        y.append(
            throughput
        )

    if not x:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x,
        y,
        marker="o",
    )

    ax.set_xlabel(
        "Elapsed time (min)"
    )

    ax.set_ylabel(
        "Completed consensus / min"
    )

    ax.set_title(
        "Consensus throughput during execution"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "consensus_throughput_timeline",
    )


def plot_scheduling_time(
    timeline,
    output_dir,
):
    x = []
    y = []
    error = []

    for row in timeline:

        elapsed = to_float(
            row["elapsed_s"]
        )

        mean = to_float(
            row[
                "scheduling_time_mean_ms"
            ]
        )

        std = to_float(
            row[
                "scheduling_time_std_ms"
            ]
        )

        if elapsed is None or mean is None:
            continue

        x.append(
            elapsed / 60.0
        )

        y.append(
            mean / 1000.0
        )

        error.append(
            (std or 0.0)
            / 1000.0
        )

    if not x:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.errorbar(
        x,
        y,
        yerr=error,
        marker="o",
        capsize=3,
    )

    ax.set_xlabel(
        "Elapsed time (min)"
    )

    ax.set_ylabel(
        "Scheduling time (s)"
    )

    ax.set_title(
        "Scheduling time during execution"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "scheduling_time_timeline",
    )


# ============================================================
# Process one swarm_report.csv
# ============================================================

def process_swarm_report(
    path,
    output_root,
    tasks_per_point,
):
    configuration = parse_configuration(
        path
    )

    if configuration is None:
        return

    rows = read_csv(path)

    if not rows:
        return

    # Keep each COMPSs execution/repetition separate.
    executions = {}

    for row in rows:

        execution_id = (
            row.get(
                "execution_id",
                ""
            ).strip()
        )

        if not execution_id:
            execution_id = "unknown"

        executions.setdefault(
            execution_id,
            [],
        ).append(row)

    for execution_id, execution_rows in executions.items():

        timeline = build_timeline(
            execution_rows,
            tasks_per_point,
        )

        if not timeline:
            continue

        queue = configuration["queue"]
        blocks = configuration["blocks"]
        agents = configuration["agents"]

        configuration_name = (
            f"q{queue}_"
            f"b{blocks}_"
            f"n{agents}"
        )

        execution_output = (
            output_root
            / configuration_name
            / execution_id
        )

        execution_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            execution_output
            / "timeline.csv",
            timeline,
        )

        plot_consensus_time(
            timeline,
            execution_output,
        )

        plot_consensus_throughput(
            timeline,
            execution_output,
        )

        plot_scheduling_time(
            timeline,
            execution_output,
        )

        print(
            f"Timeline: "
            f"{configuration_name} / "
            f"{execution_id} -> "
            f"{len(timeline)} point(s)"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate temporal SwarmTS plots "
            "for every consensus execution."
        )
    )

    parser.add_argument(
        "--results",
        default="results",
    )

    parser.add_argument(
        "--output",
        default="results/figures/timeline",
    )

    parser.add_argument(
        "--tasks-per-point",
        type=int,
        default=DEFAULT_TASKS_PER_POINT,
        help=(
            "Approximate number of completed "
            "consensus instances represented by "
            "each timeline point."
        ),
    )

    args = parser.parse_args()

    results_dir = Path(
        args.results
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    swarm_reports = sorted(
        results_dir.rglob(
            "swarm_report.csv"
        )
    )

    consensus_reports = [
        path
        for path in swarm_reports
        if "consensus" in path.parts
    ]

    print(
        f"Found {len(consensus_reports)} "
        f"consensus configuration(s)."
    )

    for path in consensus_reports:

        process_swarm_report(
            path,
            output_dir,
            args.tasks_per_point,
        )

    print()
    print(
        "Timeline figure generation complete."
    )


if __name__ == "__main__":
    main()
