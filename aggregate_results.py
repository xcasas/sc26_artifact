#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


# ============================================================
# Output columns
# ============================================================

SUMMARY_COLUMNS = [
    "scheduler",
    "queue_size",
    "blocks",
    "agents",

    "execution_samples",
    "execution_mean_s",
    "execution_variance_s2",
    "execution_std_s",

    "consensus_samples",
    "consensus_time_mean_s",
    "consensus_time_variance_s2",
    "consensus_time_std_s",

    "scheduling_samples",
    "scheduling_time_mean_s",
    "scheduling_time_variance_s2",
    "scheduling_time_std_s",

    "max_parallel_consensus",
    "max_parallel_scheduling",
]


EFFICIENCY_COLUMNS = [
    "scheduler",
    "queue_size",
    "blocks",
    "agents",
    "execution_mean_s",
    "single_agent_time_s",
    "efficiency",
]


# ============================================================
# Generic helpers
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


def mean_variance_std(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None, None, None

    mean = statistics.mean(values)

    if len(values) == 1:
        return mean, 0.0, 0.0

    variance = statistics.variance(
        values
    )

    std = statistics.stdev(
        values
    )

    return mean, variance, std


# ============================================================
# Pooled statistics
# ============================================================

def pooled_statistics(
    rows,
    count_field,
    mean_field,
    std_field,
):
    """
    Combine groups represented by:

        n_i
        mean_i
        sample_std_i

    into one global sample mean and sample variance.

    This avoids incorrectly treating every monitoring window
    as equally important.
    """

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
            (n, mean, std)
        )

    if not groups:
        return 0, None, None, None

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

    if total_n <= 1:
        return (
            total_n,
            pooled_mean,
            0.0,
            0.0,
        )

    sum_squared_deviations = 0.0

    for n, mean, std in groups:

        # Variance inside each monitoring window.
        if n > 1:
            sum_squared_deviations += (
                (n - 1) * (std ** 2)
            )

        # Difference between the window mean and global mean.
        sum_squared_deviations += (
            n * (
                mean - pooled_mean
            ) ** 2
        )

    variance = (
        sum_squared_deviations
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
# Configuration parser
# ============================================================

def parse_configuration(path: Path):
    """
    Recognizes paths such as:

        results/consensus/q10/b30/n5/gridsearch_times.csv

        results/baseline/b30/n5/gridsearch_times.csv
    """

    scheduler = None
    queue_size = None
    blocks = None
    agents = None

    for part in path.parts:

        if part == "consensus":
            scheduler = "consensus"

        elif part == "baseline":
            scheduler = "baseline"

        elif re.fullmatch(r"q\d+", part):
            queue_size = int(
                part[1:]
            )

        elif re.fullmatch(r"b\d+", part):
            blocks = int(
                part[1:]
            )

        elif re.fullmatch(r"n\d+", part):
            agents = int(
                part[1:]
            )

    if scheduler is None:
        return None

    if blocks is None:
        return None

    if agents is None:
        return None

    return {
        "scheduler": scheduler,
        "queue_size": queue_size,
        "blocks": blocks,
        "agents": agents,
    }


# ============================================================
# CSV readers
# ============================================================

def read_csv(path: Path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def read_runtime_values(path: Path):
    rows = read_csv(path)

    values = []

    for row in rows:
        value = to_float(
            row.get(
                "gridsearch_execution_time"
            )
        )

        if value is not None:
            values.append(value)

    return values


# ============================================================
# Aggregate one configuration
# ============================================================

def aggregate_configuration(
    configuration,
    runtime_rows,
    swarm_rows,
):
    execution_mean, execution_variance, execution_std = (
        mean_variance_std(
            runtime_rows
        )
    )

    (
        consensus_samples,
        consensus_mean_ms,
        consensus_variance_ms2,
        consensus_std_ms,
    ) = pooled_statistics(
        swarm_rows,
        "consensus_samples",
        "consensus_time_mean_ms",
        "consensus_time_std_ms",
    )

    (
        scheduling_samples,
        scheduling_mean_ms,
        scheduling_variance_ms2,
        scheduling_std_ms,
    ) = pooled_statistics(
        swarm_rows,
        "scheduling_samples",
        "scheduling_time_mean_ms",
        "scheduling_time_std_ms",
    )

    max_parallel_consensus_values = [
        to_int(
            row.get(
                "max_parallel_consensus"
            )
        )
        for row in swarm_rows
    ]

    max_parallel_consensus_values = [
        value
        for value in max_parallel_consensus_values
        if value is not None
    ]

    max_parallel_scheduling_values = [
        to_int(
            row.get(
                "max_parallel_scheduling"
            )
        )
        for row in swarm_rows
    ]

    max_parallel_scheduling_values = [
        value
        for value in max_parallel_scheduling_values
        if value is not None
    ]

    row = {
        "scheduler": configuration["scheduler"],
        "queue_size": (
            configuration["queue_size"]
            if configuration["queue_size"] is not None
            else ""
        ),
        "blocks": configuration["blocks"],
        "agents": configuration["agents"],

        "execution_samples": len(runtime_rows),
        "execution_mean_s": (
            execution_mean
            if execution_mean is not None
            else ""
        ),
        "execution_variance_s2": (
            execution_variance
            if execution_variance is not None
            else ""
        ),
        "execution_std_s": (
            execution_std
            if execution_std is not None
            else ""
        ),

        "consensus_samples": consensus_samples,

        "consensus_time_mean_s": (
            consensus_mean_ms / 1000.0
            if consensus_mean_ms is not None
            else ""
        ),

        "consensus_time_variance_s2": (
            consensus_variance_ms2 / 1_000_000.0
            if consensus_variance_ms2 is not None
            else ""
        ),

        "consensus_time_std_s": (
            consensus_std_ms / 1000.0
            if consensus_std_ms is not None
            else ""
        ),

        "scheduling_samples": scheduling_samples,

        "scheduling_time_mean_s": (
            scheduling_mean_ms / 1000.0
            if scheduling_mean_ms is not None
            else ""
        ),

        "scheduling_time_variance_s2": (
            scheduling_variance_ms2 / 1_000_000.0
            if scheduling_variance_ms2 is not None
            else ""
        ),

        "scheduling_time_std_s": (
            scheduling_std_ms / 1000.0
            if scheduling_std_ms is not None
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

    return row


# ============================================================
# Efficiency
# ============================================================

def build_efficiency_rows(summary_rows):
    """
    Parallel efficiency:

        E_N = T_1 / (N * T_N)

    The single-agent baseline execution for each block size
    is used as T_1.
    """

    single_agent = {}

    for row in summary_rows:
        if row["scheduler"] != "baseline":
            continue

        if int(row["agents"]) != 1:
            continue

        runtime = to_float(
            row["execution_mean_s"]
        )

        if runtime is None:
            continue

        single_agent[
            int(row["blocks"])
        ] = runtime

    rows = []

    for row in summary_rows:
        blocks = int(
            row["blocks"]
        )

        agents = int(
            row["agents"]
        )

        runtime = to_float(
            row["execution_mean_s"]
        )

        if runtime is None:
            continue

        if blocks not in single_agent:
            continue

        t1 = single_agent[
            blocks
        ]

        efficiency = (
            t1
            / (
                agents
                * runtime
            )
        )

        rows.append({
            "scheduler": row["scheduler"],
            "queue_size": row["queue_size"],
            "blocks": blocks,
            "agents": agents,
            "execution_mean_s": runtime,
            "single_agent_time_s": t1,
            "efficiency": efficiency,
        })

    return rows


# ============================================================
# CSV writer
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        if rows:
            writer.writerows(rows)

    print(
        f"Wrote: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate SC26 experiment results."
        )
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Root results directory",
    )

    parser.add_argument(
        "--output",
        default="results/processed",
        help="Output directory",
    )

    args = parser.parse_args()

    results_dir = Path(
        args.results
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    runtime_files = sorted(
        results_dir.rglob(
            "gridsearch_times.csv"
        )
    )

    configurations = defaultdict(
        lambda: {
            "runtime_values": [],
            "swarm_rows": [],
        }
    )

    for runtime_file in runtime_files:

        configuration = parse_configuration(
            runtime_file
        )

        if configuration is None:
            continue

        key = (
            configuration["scheduler"],
            configuration["queue_size"],
            configuration["blocks"],
            configuration["agents"],
        )

        runtime_values = read_runtime_values(
            runtime_file
        )

        swarm_file = (
            runtime_file.parent
            / "swarm_report.csv"
        )

        swarm_rows = read_csv(
            swarm_file
        )

        configurations[key][
            "runtime_values"
        ].extend(runtime_values)

        configurations[key][
            "swarm_rows"
        ].extend(swarm_rows)

    summary_rows = []

    for key, data in configurations.items():

        (
            scheduler,
            queue_size,
            blocks,
            agents,
        ) = key

        configuration = {
            "scheduler": scheduler,
            "queue_size": queue_size,
            "blocks": blocks,
            "agents": agents,
        }

        summary_rows.append(
            aggregate_configuration(
                configuration,
                data["runtime_values"],
                data["swarm_rows"],
            )
        )

    summary_rows.sort(
        key=lambda row: (
            row["scheduler"],
            int(
                row["queue_size"]
                or 0
            ),
            int(row["blocks"]),
            int(row["agents"]),
        )
    )

    efficiency_rows = build_efficiency_rows(
        summary_rows
    )

    efficiency_rows.sort(
        key=lambda row: (
            row["scheduler"],
            int(
                row["queue_size"]
                or 0
            ),
            int(row["blocks"]),
            int(row["agents"]),
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_output = (
        output_dir
        / "experiment_summary.csv"
    )

    efficiency_output = (
        output_dir
        / "efficiency.csv"
    )

    write_csv(
        summary_output,
        summary_rows,
        SUMMARY_COLUMNS,
    )

    write_csv(
        efficiency_output,
        efficiency_rows,
        EFFICIENCY_COLUMNS,
    )

    print()
    print(
        f"Configurations found: "
        f"{len(summary_rows)}"
    )

    if not efficiency_rows:
        print(
            "No efficiency values calculated "
            "(baseline n1 results are required)."
        )


if __name__ == "__main__":
    main()

