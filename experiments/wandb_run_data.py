import os
import sys
from typing import Callable, Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from experiments.config import WANDB_PROJECT_NAME
from utils.arg_types import str2bool, tup

import argparse
import datetime
import wandb
import tqdm
import numpy as np
import pandas as pd


ENV_CONFIGS_FILE = "door_key_change"
N_TASKS = 2
FILTER_UNCONVERGED_OUT = False
STEP_RANGE = (0, 0)

PULL_FROM_WANDB = True
DATA_FILE = "wandb_runs.pkl"

MIN_CONVERGED_RUNS = 6
CROP_END = 4000000


api = None

additional_filters = [
    {"created_at": {"$gt": datetime.datetime(2024, 2, 27, 22, 17).isoformat()}},
    # {"config.full_config.env_configs_file": ENV_CONFIGS_FILE},
    # {"config.total_time_steps": 20000000},
    # {"config.n_runs": 5},
    # {"config.total_time_steps": 10500000},
    # {"config.novelty_step": 10000000},
]

# ENV_CONFIGS_FILE = "door_key_change"
# MIN_CONVERGED_RUNS = 6
# additional_filters = [
#     {"created_at": {"$gt": datetime.datetime(2024, 2, 27, 22, 17).isoformat()}},
# ]
# CROP_END = 100000000

# ENV_CONFIGS_FILE = "simple_to_lava_crossing"
# MIN_CONVERGED_RUNS = 9
# additional_filters = [
#     {"created_at": {"$gt": datetime.datetime(2024, 2, 27, 22, 17).isoformat()}},
# ]
# CROP_END = 4000000

# ENV_CONFIGS_FILE = "lava_maze_hurt_to_safe"
# MIN_CONVERGED_RUNS = 3
# additional_filters = [
#     {"created_at": {"$gt": datetime.datetime(2024, 2, 27, 22, 17).isoformat()}},
#     {"config.novelty_step": 10000000},
# ]
# CROP_END = 1000000000

# ENV_CONFIGS_FILE = "lava_maze_safe_to_hurt"
# MIN_CONVERGED_RUNS = 2
# additional_filters = [
#     {"created_at": {"$gt": datetime.datetime(2024, 2, 27, 22, 17).isoformat()}},
#     {"config.total_time_steps": 10500000},
# ]
# CROP_END = 1000000000

ENV_CONFIGS_FILE = "walker_thigh_length"
MIN_CONVERGED_RUNS = 2
additional_filters = [
    {"created_at": {"$gt": datetime.datetime(2024, 2, 27, 22, 17).isoformat()}},
    {"config.total_time_steps": 20000000},
]
CROP_END = 10000000000


def get_api_instance():
    global api
    if api is None:
        api = wandb.Api()
    return api


def make_data_loader_parser():
    parser = argparse.ArgumentParser()

    parser.description = "Loading data script."

    parser.add_argument(
        "--wandb-project-name",
        "-wpn",
        type=str,
        default=WANDB_PROJECT_NAME,
        help="The project name to load from in wandb.",
    )
    parser.add_argument(
        "--env-configs-file",
        "-ec",
        type=str,
        default=ENV_CONFIGS_FILE,
        help="The env configs file name used in the experiments to plot.",
    )
    parser.add_argument(
        "--n-tasks",
        "-n",
        type=int,
        default=N_TASKS,
        help="The number of tasks run in these experiments. Should correspond with env configs file.",
    )

    parser.add_argument(
        "--filter-unconverged-out",
        "-fuo",
        type=str2bool,
        default=FILTER_UNCONVERGED_OUT,
        help="Whether or not to filter our the unconverged runs",
    )

    parser.add_argument(
        "--step-range",
        "-sr",
        type=tup(int),
        default=STEP_RANGE,
        help="The range of steps to include in the data.",
    )

    parser.add_argument(
        "--pull-from-wandb",
        "-pfw",
        type=str2bool,
        default=PULL_FROM_WANDB,
    )
    parser.add_argument("--data-file", "-df", type=str, default=DATA_FILE)

    return parser


def edit_runs(
    project_name: str, update_run: Callable[[Any], None], include_sweeps: bool = False
):
    api = get_api_instance()

    wandb_runs = api.runs(
        project_name,
        filters={
            "$and": [
                {"state": "finished"},
                *additional_filters,
            ]
        },
        include_sweeps=include_sweeps,
    )

    for wandb_run in tqdm.tqdm(wandb_runs):
        update_run(wandb_run)
        wandb_run.update()


def load_data(args: argparse.Namespace) -> pd.DataFrame:

    data_file_path = f"./data/{args.data_file}"

    if not args.pull_from_wandb and os.path.exists(data_file_path):
        return pd.read_pickle(data_file_path)

    api = get_api_instance()

    wandb_runs = api.runs(
        args.wandb_project_name,
        filters={
            "$and": [
                {"config.full_config.env_configs_file": args.env_configs_file},
                {"state": "finished"},
                *additional_filters,
            ]
            + [
                {"tags": {"$in": [f"converged_{n}"]}}
                for n in range(args.n_tasks)
                if args.filter_unconverged_out
            ]
            + [
                {
                    "$or": [
                        {"config.archived": {"$exists": False}},
                        {"config.archived": False},
                    ]
                }
            ],
        },
        include_sweeps=False,
    )

    print("Loading data . . .")

    mapping_by_experiment_name = {}

    for wandb_run in tqdm.tqdm(wandb_runs):
        experiment_name = wandb_run.config["experiment_name"]
        run_id = wandb_run.id

        df = pd.DataFrame(
            wandb_run.scan_history(keys=["global_step", "rollout/ep_rew_mean"])
        )
        df["global_step"] = df["global_step"].astype(int)
        df["run_id"] = run_id
        df["experiment_name"] = experiment_name

        converged = []

        for i in range(args.n_tasks):
            converged.append(f"converged_{i}" in wandb_run.tags)
            df[f"converged_{i}"] = converged[i]

        df[f"converged_all"] = np.all(converged)

        df["novelty_step"] = wandb_run.config["novelty_step"]
        df["n_tasks"] = wandb_run.config["n_tasks"]

        if experiment_name not in mapping_by_experiment_name:
            mapping_by_experiment_name[experiment_name] = {}

        mapping_by_experiment_name[experiment_name][run_id] = df

    experiment_names, mapping_by_run_id = map(
        list, zip(*mapping_by_experiment_name.items())
    )

    dfs_by_experiment_name = []
    for m in mapping_by_run_id:
        run_ids, experiment_dfs = map(list, zip(*m.items()))
        dfs_by_experiment_name.append(pd.concat(experiment_dfs, keys=run_ids))

    full_df = pd.concat(
        dfs_by_experiment_name,
        keys=experiment_names,
        names=["experiment_name_idx", "run_id_idx", "row_idx"],
    )

    if args.step_range[0] > 0 and args.step_range[1] > 0:
        full_df = full_df[
            full_df["global_step"].between(args.step_range[0], args.step_range[1])
        ]
    elif args.step_range[0] > 0:
        full_df = full_df[full_df["global_step"].gt(args.step_range[0])]
    elif args.step_range[1] > 0:
        full_df = full_df[full_df["global_step"].lt(args.step_range[1])]

    full_df = full_df.sort_index()

    os.makedirs("./data", exist_ok=True)
    full_df.to_pickle(data_file_path)

    return full_df


def get_index_dict(df: pd.DataFrame):
    experiment_name_idx = df.index.get_level_values("experiment_name_idx")
    run_id_idx = df.index.get_level_values("run_id_idx")
    row_idx = df.index.get_level_values("row_idx")

    indices = {}

    for experiment_name, run_id, row_num in zip(
        experiment_name_idx, run_id_idx, row_idx
    ):
        if experiment_name not in indices:
            indices[experiment_name] = {}
        if run_id not in indices[experiment_name]:
            indices[experiment_name][run_id] = []
        indices[experiment_name][run_id].append(row_num)

    return indices
