#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ==========================
# Per-task swarm metrics
# ==========================

TASK_RE = re.compile(r"^Task:\s*(.+?)\s*$")

RETRIES_RE = re.compile(
    r"^Retries:\s*(-?\d+)\s*$"
)

TIMEOUT_RETRIES_RE = re.compile(
    r"^Timeout retries:\s*(-?\d+)\s*$"
)

PREPARE_CONFLICTS_RE = re.compile(
    r"^Prepare conflicts:\s*(-?\d+)\s*$"
)

COMMIT_CONFLICTS_RE = re.compile(
    r"^Commit conflicts:\s*(-?\d+)\s*$"
)

CONSENSUS_TIME_RE = re.compile(
    r"^Consensus time:\s*(-?\d+)\s*$"
)

SCHEDULING_TIME_RE = re.compile(
    r"^Scheduling time:\s*(-?\d+)\s*$"
)

COMMIT_TO_START_TIME_RE = re.compile(
    r"^Commit-to-start time:\s*(-?\d+)\s*$"
)

EXECUTION_TIME_RE = re.compile(
    r"^Execution time:\s*(-?\d+)\s*$"
)

TERMINATION_TIME_RE = re.compile(
    r"^Termination time:\s*(-?\d+)\s*$"
)

NEW_TASK_MESSAGES_RE = re.compile(
    r"^NEW_TASK messages:\s*(-?\d+)\s*$"
)

PROPOSE_MESSAGES_RE = re.compile(
    r"^PROPOSE(?: received)? messages:\s*(-?\d+)\s*$"
)

PREPARE_MESSAGES_RE = re.compile(
    r"^PREPARE(?: received)? messages:\s*(-?\d+)\s*$"
)

COMMIT_MESSAGES_RE = re.compile(
    r"^COMMIT(?: received)? messages:\s*(-?\d+)\s*$"
)

TASK_STARTED_MESSAGES_RE = re.compile(
    r"^TASK_STARTED messages:\s*(-?\d+)\s*$"
)

TASK_FINISHED_MESSAGES_RE = re.compile(
    r"^TASK_FINISHED messages:\s*(-?\d+)\s*$"
)


# ==========================
# GridSearch runtime
# ==========================

GRIDSEARCH_TIME_RE = re.compile(
    r"GridSearch execution time:\s*([0-9]+(?:\.[0-9]+)?)"
)


# ==========================
# CSV definitions
# ==========================

SWARM_COLUMNS = [
    "source_file",
    "source_host",
    "execution_id",
    "task_id",
    "retries",
    "timeout_retries",
    "prepare_conflicts",
    "commit_conflicts",
    "consensus_time",
    "scheduling_time",
    "commit_to_start_time",
    "execution_time",
    "termination_time",
    "new_task_messages",
    "propose_messages",
    "prepare_messages",
    "commit_messages",
    "task_started_messages",
    "task_finished_messages",
]

GRIDSEARCH_COLUMNS = [
    "execution_id",
    "source_file",
    "gridsearch_execution_time",
]


# ==========================
# Arguments
# ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically extract SwarmTS task metrics and "
            "GridSearch execution times from all experiment logs."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Root results directory. Default: results",
    )

    return parser.parse_args()


# ==========================
# Helpers
# ==========================

def infer_source_host(source_file: Path) -> str:
    name = source_file.name

    if name.endswith(".OUT"):
        return name[:-4]

    return source_file.stem


def infer_execution_id(path: Path) -> str:
    parts = path.resolve().parts

    if ".COMPSs" in parts:
        index = parts.index(".COMPSs")

        if len(parts) > index + 1:
            return parts[index + 1]

    return ""


def find_experiment_logs(
    results_dir: Path,
) -> List[Path]:

    if not results_dir.exists():
        return []

    return sorted(
        path
        for path in results_dir.rglob("logs")
        if path.is_dir()
    )


def collect_out_files(
    logs_dir: Path,
) -> List[Path]:

    return sorted(
        path
        for path in logs_dir.rglob("*.OUT")
        if path.is_file()
    )


# ==========================
# Task parsing
# ==========================

def new_record(
    source_file: Path,
    task_id: str,
) -> Dict[str, str]:

    return {
        "source_file": str(source_file),
        "source_host": infer_source_host(
            source_file
        ),
        "execution_id": infer_execution_id(
            source_file
        ),
        "task_id": task_id,
        "retries": "",
        "timeout_retries": "",
        "prepare_conflicts": "",
        "commit_conflicts": "",
        "consensus_time": "",
        "scheduling_time": "",
        "commit_to_start_time": "",
        "execution_time": "",
        "termination_time": "",
        "new_task_messages": "",
        "propose_messages": "",
        "prepare_messages": "",
        "commit_messages": "",
        "task_started_messages": "",
        "task_finished_messages": "",
    }


def record_has_report_values(
    record: Dict[str, str],
) -> bool:

    metadata = {
        "source_file",
        "source_host",
        "execution_id",
        "task_id",
    }

    for column in SWARM_COLUMNS:
        if (
            column not in metadata
            and record[column] != ""
        ):
            return True

    return False


def apply_matchers(
    current: Dict[str, str],
    line: str,
) -> bool:

    matchers = (
        (RETRIES_RE, "retries"),
        (
            TIMEOUT_RETRIES_RE,
            "timeout_retries",
        ),
        (
            PREPARE_CONFLICTS_RE,
            "prepare_conflicts",
        ),
        (
            COMMIT_CONFLICTS_RE,
            "commit_conflicts",
        ),
        (
            CONSENSUS_TIME_RE,
            "consensus_time",
        ),
        (
            SCHEDULING_TIME_RE,
            "scheduling_time",
        ),
        (
            COMMIT_TO_START_TIME_RE,
            "commit_to_start_time",
        ),
        (
            EXECUTION_TIME_RE,
            "execution_time",
        ),
        (
            TERMINATION_TIME_RE,
            "termination_time",
        ),
        (
            NEW_TASK_MESSAGES_RE,
            "new_task_messages",
        ),
        (
            PROPOSE_MESSAGES_RE,
            "propose_messages",
        ),
        (
            PREPARE_MESSAGES_RE,
            "prepare_messages",
        ),
        (
            COMMIT_MESSAGES_RE,
            "commit_messages",
        ),
        (
            TASK_STARTED_MESSAGES_RE,
            "task_started_messages",
        ),
        (
            TASK_FINISHED_MESSAGES_RE,
            "task_finished_messages",
        ),
    )

    for matcher, field in matchers:
        match = matcher.match(line)

        if match:
            current[field] = match.group(1)
            return True

    return False


def parse_out_file(
    path: Path,
) -> List[Dict[str, str]]:

    records: List[Dict[str, str]] = []

    current: Optional[
        Dict[str, str]
    ] = None

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for raw_line in handle:
            line = raw_line.strip()

            task_match = TASK_RE.match(
                line
            )

            if task_match:

                if (
                    current is not None
                    and record_has_report_values(
                        current
                    )
                ):
                    records.append(current)

                current = new_record(
                    path,
                    task_match.group(1),
                )

                continue

            if current is None:
                continue

            apply_matchers(
                current,
                line,
            )

    if (
        current is not None
        and record_has_report_values(
            current
        )
    ):
        records.append(current)

    return records


# ==========================
# GridSearch timing parsing
# ==========================

def parse_gridsearch_times(
    logs_dir: Path,
) -> List[Dict[str, str]]:

    rows: List[Dict[str, str]] = []

    for path in logs_dir.rglob(
        "job1_NEW.out"
    ):
        if not path.is_file():
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:

                for line in handle:

                    match = (
                        GRIDSEARCH_TIME_RE.search(
                            line
                        )
                    )

                    if not match:
                        continue

                    rows.append(
                        {
                            "execution_id":
                                infer_execution_id(
                                    path
                                ),
                            "source_file":
                                str(path),
                            "gridsearch_execution_time":
                                match.group(1),
                        }
                    )

        except OSError as exc:
            print(
                f"warning: could not read "
                f"{path}: {exc}",
                file=sys.stderr,
            )

    return rows


# ==========================
# CSV output
# ==========================

def write_csv(
    path: Path,
    columns: List[str],
    rows: List[Dict[str, str]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        writer.writerows(rows)


# ==========================
# Process one experiment
# ==========================

def process_experiment(
    logs_dir: Path,
) -> None:

    experiment_dir = logs_dir.parent

    swarm_output = (
        experiment_dir
        / "swarm_report.csv"
    )

    times_output = (
        experiment_dir
        / "gridsearch_times.csv"
    )

    out_files = collect_out_files(
        logs_dir
    )

    swarm_rows: List[
        Dict[str, str]
    ] = []

    for out_file in out_files:
        swarm_rows.extend(
            parse_out_file(
                out_file
            )
        )

    gridsearch_rows = (
        parse_gridsearch_times(
            logs_dir
        )
    )

    write_csv(
        swarm_output,
        SWARM_COLUMNS,
        swarm_rows,
    )

    write_csv(
        times_output,
        GRIDSEARCH_COLUMNS,
        gridsearch_rows,
    )

    print(
        f"{experiment_dir}: "
        f"{len(swarm_rows)} task rows, "
        f"{len(gridsearch_rows)} "
        f"GridSearch time(s)"
    )


# ==========================
# Main
# ==========================

def main() -> int:
    args = parse_args()

    results_dir = (
        args.results
        .expanduser()
        .resolve()
    )

    logs_dirs = (
        find_experiment_logs(
            results_dir
        )
    )

    if not logs_dirs:
        print(
            f"No experiment log "
            f"directories found under "
            f"{results_dir}",
            file=sys.stderr,
        )

        return 1

    print(
        f"Found {len(logs_dirs)} "
        f"experiment(s)."
    )

    print()

    for logs_dir in logs_dirs:
        process_experiment(
            logs_dir
        )

    print()
    print(
        "Result extraction complete."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
