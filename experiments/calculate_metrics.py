import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import pandas as pd

from utils.args import get_args
import experiments.wandb_run_data as wrd
from experiments.wandb_run_data import (
    make_data_loader_parser,
    load_data,
    get_index_dict,
)

wrd.PULL_FROM_WANDB = False
WINDOW_SIZE = 50
CONVERGENCE_REWARD_THRESHOLD = 0.9


def make_parser():
    parser = make_data_loader_parser()

    parser.add_argument(
        "--window-size",
        "-es",
        type=int,
        default=WINDOW_SIZE,
        help="The span to use for the calculation of the ema.",
    )

    parser.add_argument(
        "--convergence-reward-threshold",
        "-crt",
        type=float,
        default=CONVERGENCE_REWARD_THRESHOLD,
        help="The convergence reward threshold.",
    )

    return parser


def calculate_metrics(args: argparse.Namespace, df: pd.DataFrame):
    index_dict = get_index_dict(df=df)

    metrics_df = pd.DataFrame()

    for experiment_name in index_dict:
        for run_id in index_dict[experiment_name]:
            rewards_df = (
                df.loc[experiment_name, run_id]
                .set_index("global_step", drop=True)
                .drop(columns=["run_id", "experiment_name"])
                .rename(columns={"rollout/ep_rew_mean": "ep_reward"})
            )

            rewards_df["ep_reward_smoothed"] = (
                rewards_df["ep_reward"].rolling(window=args.window_size).mean()
            )

            rewards_df["converged"] = (
                rewards_df["ep_reward_smoothed"] > args.convergence_reward_threshold
            )

            rewards_df["task_num"] = (
                rewards_df["converged"].astype(int).diff() < 0
            ).cumsum()

            rewards_df["first_converged"] = (
                rewards_df["converged"].astype(int).diff() > 0
            )

            steps_to_converge_per_task = (
                rewards_df["first_converged"][rewards_df["first_converged"] == True]
                .index.to_series()
                .diff()
            )
            steps_to_converge_per_task.iloc[0] = rewards_df["first_converged"][
                rewards_df["first_converged"] == True
            ].index[0]
            steps_to_converge_per_task.index = range(len(steps_to_converge_per_task))

            for i in steps_to_converge_per_task.index:
                metrics_df.loc[f"converged_step_{i}", (experiment_name, run_id)] = (
                    steps_to_converge_per_task[i]
                )

        import pdb

        pdb.set_trace()

        return metrics_df


def main(args):
    df = load_data(args=args)
    metrics_df = calculate_metrics(args=args, df=df)
    metrics_df.to_pickle(f"./data/metrics.pkl")


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser)
    main(args=args)
