import os
import sys
import time
import pickle

import numpy as np

from sklearn import datasets
from sklearn.utils import shuffle

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.model_selection import GridSearchCV

from pycompss.api.task import task
from pycompss.api.api import compss_barrier


def csvm_accuracy_scorer(clf, x_score, y_real):
    return clf.score(x_score, y_real)


def assert_successful_fitting(searcher, n_splits, *attributes):
    attributes = list(attributes)

    if getattr(searcher, "refit", True) is False:
        unavailable = {"best_estimator_"}

        if getattr(searcher, "multimetric_", False):
            unavailable.update(
                {"best_score_", "best_params_", "best_index_"}
            )

        attributes = [
            att for att in attributes
            if att not in unavailable
        ]


def create_param_grid(combinations):
    """
    Create the hyperparameter grid.

    Supported configurations:
      1  combination  -> 1 x 1
      9  combinations -> 3 x 3
      25 combinations -> 5 x 5
    """

    if combinations == 1:
        gamma = [0.1]
        c = [0.1]

    elif combinations == 9:
        gamma = [0.1, 0.2, 0.3]
        c = [0.1, 0.2, 0.3]

    elif combinations == 25:
        gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
        c = [0.1, 0.2, 0.3, 0.4, 0.5]

    else:
        raise ValueError(
            f"Unsupported number of combinations: {combinations}. "
            "Supported values are 1, 9, and 25."
        )

    return {
        "gamma": gamma,
        "c": c,
    }


def launch_grid_search(x, y, n_splits, param_grid):
    print("Preparing CSVM", flush=True)

    csvm = CascadeSVM(
        check_convergence=False,
        max_iter=2,
        random_state=0,
    )

    print("Preparing GridSearch", flush=True)

    searcher = GridSearchCV(
        csvm,
        param_grid,
        cv=n_splits,
        scoring=csvm_accuracy_scorer,
        nested=True,
        refit=False,
    )

    print("Starting search...", flush=True)

    t0 = time.time()

    searcher.fit(x, y)
    compss_barrier()

    t1 = time.time()

    print("Search completed", flush=True)

    assert_successful_fitting(
        searcher,
        n_splits,
        "best_estimator_",
        "best_score_",
        "best_params_",
        "best_index_",
        "scorer_",
    )

    return t1 - t0


def load_dataset(dataset, num_sample_blocks):
    print("Loading dataset", flush=True)

    if dataset.upper() == "IRIS":
        print("Dataset: IRIS", flush=True)

        x_np, y_np = datasets.load_iris(return_X_y=True)
        x_np, y_np = shuffle(x_np, y_np, random_state=0)

        x = ds.array(
            x_np,
            block_size=(60, 4),
        )

        y = ds.array(
            y_np[:, np.newaxis],
            block_size=(60, 1),
        )

    elif dataset.upper() == "DIGITS":
        print("Dataset: DIGITS", flush=True)

        x_np, y_np = datasets.load_digits(return_X_y=True)
        x_np, y_np = shuffle(x_np, y_np, random_state=0)

        x = ds.array(
            x_np,
            block_size=(50, 64),
        )

        y = ds.array(
            y_np[:, np.newaxis],
            block_size=(50, 1),
        )

    else:
        if dataset.upper() == "AT":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(
                script_dir,
                "7500_100.pickle",
            )
        else:
            dataset_path = dataset

        print(
            f"Dataset: {dataset_path}",
            flush=True,
        )

        with open(dataset_path, "rb") as data:
            x_np = pickle.load(data)
            y_np = pickle.load(data)

        num_samples = len(x_np)
        num_features = len(x_np[0])

        samples_per_block = (
            num_samples + num_sample_blocks - 1
        ) // num_sample_blocks

        features_per_block = num_features

        print(
            f"Dataset with {num_samples} instances "
            f"and {num_features} features",
            flush=True,
        )

        print(
            f"Number of sample blocks: {num_sample_blocks}",
            flush=True,
        )

        print(
            f"Block size: "
            f"{samples_per_block}x{features_per_block}",
            flush=True,
        )

        x = ds.array(
            x_np,
            block_size=(
                samples_per_block,
                features_per_block,
            ),
        )

        y = ds.array(
            y_np,
            block_size=(
                samples_per_block,
                1,
            ),
        )

    print(
        f"Dataset with {len(x_np)} instances "
        f"and {len(x_np[0])} features",
        flush=True,
    )

    return x, y


def main(
    combinations=9,
    num_sample_blocks=30,
    num_gridsearch=1,
    dataset="IRIS",
):
    combinations = int(combinations)
    num_sample_blocks = int(num_sample_blocks)
    num_gridsearch = int(num_gridsearch)

    print(
        f"Configuration: "
        f"combinations={combinations}, "
        f"blocks={num_sample_blocks}, "
        f"gridsearches={num_gridsearch}, "
        f"dataset={dataset}",
        flush=True,
    )

    x, y = load_dataset(
        dataset,
        num_sample_blocks,
    )

    print(
        f"Creating hyperparameter grid "
        f"with {combinations} combinations",
        flush=True,
    )

    param_grid = create_param_grid(
        combinations,
    )

    actual_combinations = (
        len(param_grid["gamma"])
        * len(param_grid["c"])
    )

    print(
        f"Actual number of hyperparameter "
        f"combinations: {actual_combinations}",
        flush=True,
    )

    print(
        f"Parameter grid: {param_grid}",
        flush=True,
    )

    n_splits = 4

    for run in range(num_gridsearch):
        print(
            f"GridSearch run "
            f"{run + 1}/{num_gridsearch}",
            flush=True,
        )

        execution_time = launch_grid_search(
            x,
            y,
            n_splits,
            param_grid,
        )

        print(
            f"GridSearch execution time: "
            f"{execution_time}",
            flush=True,
        )


@task()
def main_agents(
    combinations="9",
    num_sample_blocks="30",
    num_gridsearch="1",
    dataset="IRIS",
):
    main(
        combinations=int(combinations),
        num_sample_blocks=int(num_sample_blocks),
        num_gridsearch=int(num_gridsearch),
        dataset=dataset,
    )


if __name__ == "__main__":
    combinations = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 9
    )

    num_sample_blocks = (
        int(sys.argv[2])
        if len(sys.argv) > 2
        else 30
    )

    num_gridsearch = (
        int(sys.argv[3])
        if len(sys.argv) > 3
        else 1
    )

    dataset = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "IRIS"
    )

    main(
        combinations,
        num_sample_blocks,
        num_gridsearch,
        dataset,
    )
