import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import wandb
import tqdm
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"figure.figsize": (30, 15)})


from utils.arg_types import str2bool, tup
from utils.args import get_args
from config import WANDB_PROJECT_NAME

ENV_CONFIGS_FILE = "door_key_change"
N_TASKS = 2
FILTER_UNCONVERGED_OUT = True
IMG_NAME = "converged_ep_rew_mean.png"
ESTIMATOR = "mean"
ERROR_BAR_TYPE = "ci"
ERROR_BAR_ARG = 95
STEP_RANGE = (0, 0)


def estimator_type(s: str):
    if s.startswith("iq_"):

        def f(df: pd.Series):
            return df[df.between(*df.quantile([0.25, 0.75]))].agg(s[3:])

        return f
    return s


def make_error_bar_arg(error_bar_type: str, error_bar_arg: float):
    if error_bar_type == "iq_sd":

        def f(df: pd.Series):
            mu = df.between(*df.quantile([0.25, 0.75])).mean()
            sigma = df.between(*df.quantile([0.25, 0.75])).std()
            return mu - error_bar_arg * sigma, mu + error_bar_arg * sigma

        return f
    return (error_bar_type, error_bar_arg)


def make_parser():
    parser = argparse.ArgumentParser()

    parser.description = (
        "A python script to pull the reward data from wandb and plot it using seaborn."
    )

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
        "--img-name",
        "-i",
        type=str,
        default=IMG_NAME,
        help="The name of the image to save the plot to in the figures folder.",
    )

    parser.add_argument(
        "--estimator",
        "-e",
        type=estimator_type,
        default=ESTIMATOR,
        help="The estimator to use (in seaborns lineplot function) to aggregate data.",
    )

    parser.add_argument(
        "--error-bar-type",
        "-ebt",
        type=str,
        default=ERROR_BAR_TYPE,
        help="The type of error bar (in seaborns lineplot function) to use.",
    )
    parser.add_argument(
        "--error-bar-arg",
        "-eba",
        type=float,
        default=ERROR_BAR_ARG,
        help="The type of error bar argument (in seaborns lineplot function) to use.",
    )

    parser.add_argument(
        "--step-range",
        "-sr",
        type=tup(int),
        default=STEP_RANGE,
        help="The range of steps to include in the data.",
    )

    return parser


def load_data(args: argparse.Namespace) -> pd.DataFrame:

    api = wandb.Api()

    wandb_runs = api.runs(
        args.wandb_project_name,
        filters={
            "$and": [
                {"config.full_config.env_configs_file": args.env_configs_file},
            ]
            + [
                {"tags": {"$in": [f"converged_{n}"]}}
                for n in range(args.n_tasks)
                if args.filter_unconverged_out
            ]
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

    return full_df


def visualize_data(args: argparse.Namespace, df: pd.DataFrame) -> None:
    plot = sns.lineplot(
        x="global_step",
        y="rollout/ep_rew_mean",
        hue="experiment_name",
        data=df,
        errorbar=make_error_bar_arg(args.error_bar_type, args.error_bar_arg),
        estimator=args.estimator,
    )
    plot.figure.savefig(f"figures/{args.img_name}")


def main(args):
    df = load_data(args)
    visualize_data(args, df)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser, configs_root="./configs")
    main(args)
