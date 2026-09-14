#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


CONSENSUS_START = re.compile(r"^Consensus window report$")
SCHEDULING_START = re.compile(r"^Scheduling window report$")

WINDOW_START = re.compile(r"^Window start:\s*(\d+)")
WINDOW_END = re.compile(r"^Window end:\s*(\d+)")

COMPLETED_CONSENSUS = re.compile(r"^Completed consensus:\s*(\d+)")
MEAN_CONSENSUS = re.compile(
    r"^Mean consensus time:\s*([0-9.eE+-]+)\s*ms"
)
STD_CONSENSUS = re.compile(
    r"^Sample consensus standard deviation:\s*([0-9.eE+-]+)\s*ms"
)
CURRENT_PARALLEL_CONSENSUS = re.compile(
    r"^Current parallel consensus:\s*(\d+)"
)
MAX_PARALLEL_CONSENSUS = re.compile(
    r"^Max parallel consensus in window:\s*(\d+)"
)

COMPLETED_SCHEDULING = re.compile(r"^Completed scheduling:\s*(\d+)")
MEAN_SCHEDULING = re.compile(
    r"^Mean scheduling time:\s*([0-9.eE+-]+)\s*ms"
)
STD_SCHEDULING = re.compile(
    r"^Sample scheduling standard deviation:\s*([0-9.eE+-]+)\s*ms"
)
CURRENT_PARALLEL_SCHEDULING = re.compile(
    r"^Current parallel scheduling:\s*(\d+)"
)
MAX_PARALLEL_SCHEDULING = re.compile(
    r"^Max parallel scheduling in window:\s*(\d+)"
)

GRIDSEARCH_TIME = re.compile(
    r"GridSearch execution time:\s*([0-9.eE+-]+)"
)


SWARM_COLUMNS = [
    "execution_id",
    "source_host",
    "source_file",
    "window_index",

    "consensus_window_start_ms",
    "consensus_window_end_ms",
    "consensus_samples",
    "consensus_time_mean_ms",
    "consensus_time_std_ms",
    "current_parallel_consensus",
    "max_parallel_consensus",

    "scheduling_window_start_ms",
    "scheduling_window_end_ms",
    "scheduling_samples",
    "scheduling_time_mean_ms",
    "scheduling_time_std_ms",
    "current_parallel_scheduling",
    "max_parallel_scheduling",
]


GRIDSEARCH_COLUMNS = [
    "execution_id",
    "source_file",
    "gridsearch_execution_time",
]


def infer_execution_id(path):
    parts = path.parts

    try:
        index = parts.index(".COMPSs")
        if index + 1 < len(parts):
            return parts[index + 1]
    except ValueError:
        pass

    return ""


def infer_host(path):
    name = path.name

    # Example:
    # gs08r2b04-ib0.OUT -> gs08r2b04-ib0
    if name.endswith(".OUT"):
        return name[:-4]

    return path.stem


def parse_consensus_block(lines, start):
    result = {
        "consensus_window_start_ms": "",
        "consensus_window_end_ms": "",
        "consensus_samples": "",
        "consensus_time_mean_ms": "",
        "consensus_time_std_ms": "",
        "current_parallel_consensus": "",
        "max_parallel_consensus": "",
    }

    i = start + 1

    while i < len(lines):
        line = lines[i].strip()

        if (
            CONSENSUS_START.match(line)
            or SCHEDULING_START.match(line)
        ):
            break

        match = WINDOW_START.match(line)
        if match:
            result["consensus_window_start_ms"] = match.group(1)

        match = WINDOW_END.match(line)
        if match:
            result["consensus_window_end_ms"] = match.group(1)

        match = COMPLETED_CONSENSUS.match(line)
        if match:
            result["consensus_samples"] = match.group(1)

        match = MEAN_CONSENSUS.match(line)
        if match:
            result["consensus_time_mean_ms"] = match.group(1)

        match = STD_CONSENSUS.match(line)
        if match:
            result["consensus_time_std_ms"] = match.group(1)

        match = CURRENT_PARALLEL_CONSENSUS.match(line)
        if match:
            result["current_parallel_consensus"] = match.group(1)

        match = MAX_PARALLEL_CONSENSUS.match(line)
        if match:
            result["max_parallel_consensus"] = match.group(1)

        i += 1

    return result, i


def parse_scheduling_block(lines, start):
    result = {
        "scheduling_window_start_ms": "",
        "scheduling_window_end_ms": "",
        "scheduling_samples": "",
        "scheduling_time_mean_ms": "",
        "scheduling_time_std_ms": "",
        "current_parallel_scheduling": "",
        "max_parallel_scheduling": "",
    }

    i = start + 1

    while i < len(lines):
        line = lines[i].strip()

        if (
            CONSENSUS_START.match(line)
            or SCHEDULING_START.match(line)
        ):
            break

        match = WINDOW_START.match(line)
        if match:
            result["scheduling_window_start_ms"] = match.group(1)

        match = WINDOW_END.match(line)
        if match:
            result["scheduling_window_end_ms"] = match.group(1)

        match = COMPLETED_SCHEDULING.match(line)
        if match:
            result["scheduling_samples"] = match.group(1)

        match = MEAN_SCHEDULING.match(line)
        if match:
            result["scheduling_time_mean_ms"] = match.group(1)

        match = STD_SCHEDULING.match(line)
        if match:
            result["scheduling_time_std_ms"] = match.group(1)

        match = CURRENT_PARALLEL_SCHEDULING.match(line)
        if match:
            result["current_parallel_scheduling"] = match.group(1)

        match = MAX_PARALLEL_SCHEDULING.match(line)
        if match:
            result["max_parallel_scheduling"] = match.group(1)

        i += 1

    return result, i


def parse_swarm_file(path):
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return []

    lines = text.splitlines()

    consensus_blocks = []
    scheduling_blocks = []

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if CONSENSUS_START.match(line):
            block, i = parse_consensus_block(lines, i)
            consensus_blocks.append(block)
            continue

        if SCHEDULING_START.match(line):
            block, i = parse_scheduling_block(lines, i)
            scheduling_blocks.append(block)
            continue

        i += 1

    rows = []

    number_windows = max(
        len(consensus_blocks),
        len(scheduling_blocks),
    )

    execution_id = infer_execution_id(path)
    host = infer_host(path)

    for index in range(number_windows):
        row = {
            column: ""
            for column in SWARM_COLUMNS
        }

        row["execution_id"] = execution_id
        row["source_host"] = host
        row["source_file"] = str(path)
        row["window_index"] = index

        if index < len(consensus_blocks):
            row.update(consensus_blocks[index])

        if index < len(scheduling_blocks):
            row.update(scheduling_blocks[index])

        rows.append(row)

    return rows


def parse_gridsearch_file(path):
    rows = []

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return rows

    execution_id = infer_execution_id(path)

    for match in GRIDSEARCH_TIME.finditer(text):
        rows.append({
            "execution_id": execution_id,
            "source_file": str(path),
            "gridsearch_execution_time": match.group(1),
        })

    return rows


def write_csv(path, rows, columns):
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
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def find_experiment_logs(results_dir):
    return sorted(
        path
        for path in results_dir.rglob("logs")
        if path.is_dir()
    )


def collect_out_files(logs_dir):
    return sorted(
        path
        for path in logs_dir.rglob("*.OUT")
        if path.is_file()
    )


def process_experiment(logs_dir):
    experiment_dir = logs_dir.parent

    swarm_rows = []
    gridsearch_rows = []

    out_files = collect_out_files(logs_dir)

    for path in out_files:
        swarm_rows.extend(
            parse_swarm_file(path)
        )

        gridsearch_rows.extend(
            parse_gridsearch_file(path)
        )

    swarm_output = (
        experiment_dir / "swarm_report.csv"
    )

    runtime_output = (
        experiment_dir / "gridsearch_times.csv"
    )

    write_csv(
        swarm_output,
        swarm_rows,
        SWARM_COLUMNS,
    )

    write_csv(
        runtime_output,
        gridsearch_rows,
        GRIDSEARCH_COLUMNS,
    )

    print(
        f"{experiment_dir}: "
        f"{len(swarm_rows)} scheduler windows, "
        f"{len(gridsearch_rows)} GridSearch time(s)"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        default="results",
        help="Root results directory",
    )

    args = parser.parse_args()

    results_dir = Path(
        args.results
    ).resolve()

    if not results_dir.exists():
        print(
            f"Results directory does not exist: "
            f"{results_dir}"
        )
        return

    experiment_logs = find_experiment_logs(
        results_dir
    )

    print(
        f"Found {len(experiment_logs)} "
        f"experiment(s).\n"
    )

    for logs_dir in experiment_logs:
        process_experiment(logs_dir)

    print("\nResult extraction complete.")


if __name__ == "__main__":
    main()
