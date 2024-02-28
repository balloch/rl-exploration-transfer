import os
import sys
from typing import Dict, Callable, List, Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import json

from utils.args import get_args
import experiments.wandb_run_data as wrd
from experiments.wandb_run_data import (
    make_data_loader_parser,
    load_data,
    get_index_dict,
)

wrd.PULL_FROM_WANDB = False
wrd.FILTER_UNCONVERGED_OUT = False
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


def calculate_metrics(
    args: argparse.Namespace,
    df: pd.DataFrame,
    metrics_fns: Dict[str, Callable[[pd.DataFrame], Any]],
) -> Dict[str, Dict[str, List[Any]]]:
    index_dict = get_index_dict(df=df)

    metrics = {}

    for experiment_name in index_dict:
        metrics[experiment_name] = {k: [] for k in metrics_fns}
        for run_id in index_dict[experiment_name]:
            rewards_df = (
                df.loc[experiment_name, run_id]
                .set_index("global_step", drop=True)
                .drop(columns=["run_id", "experiment_name"])
                .rename(columns={"rollout/ep_rew_mean": "rewards"})
            )

            [
                metrics[experiment_name][k].append(v(rewards_df))
                for k, v in metrics_fns.items()
            ]
        for k in metrics_fns:
            metrics[experiment_name][k] = np.array(metrics[experiment_name][k])

    return metrics


def aggregate_metrics(
    metrics: Dict[str, Dict[str, np.ndarray]],
    aggs: Dict[str, Callable[[Dict[str, np.ndarray]], Any]],
):
    return {
        experiment_name: {agg_name: agg(metrics_dict) for agg_name, agg in aggs.items()}
        for experiment_name, metrics_dict in metrics.items()
    }


def iq(arr: np.ndarray):
    sorted_arr = np.sort(arr)
    n = len(arr)
    return sorted_arr[int(np.ceil(0.25 * n)) : int(np.floor(0.75 * n)) + 1]


def bootstrapped_sampling(arr, k=6, m=5):
    n = len(arr)
    idx = np.array([np.random.choice(n, size=(k,), replace=False) for _ in range(m)])
    return arr[idx]


class Aggregators:

    @staticmethod
    def converged(metrics: Dict[str, np.ndarray]):
        return {
            k: {
                "mean": metrics[k][metrics["converged"]].mean(),
                "std": metrics[k][metrics["converged"]].std(),
            }
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def converged_iq(metrics: Dict[str, np.ndarray]):
        return {
            k: {
                "mean": iq(metrics[k][metrics["converged"]]).mean(),
                "std": iq(metrics[k][metrics["converged"]]).std(),
            }
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def bootstrapped_converged(metrics: Dict[str, np.ndarray]):
        return {
            k: {
                "mean": bootstrapped_sampling(metrics[k][metrics["converged"]])
                .mean(axis=1)
                .mean(),
                "std": bootstrapped_sampling(metrics[k][metrics["converged"]])
                .std(axis=1)
                .mean(),
            }
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def bootstrapped_converged_iq(metrics: Dict[str, np.ndarray]):
        return {
            k: {
                "mean": iq(bootstrapped_sampling(metrics[k][metrics["converged"]]))
                .mean(axis=1)
                .mean(),
                "std": iq(bootstrapped_sampling(metrics[k][metrics["converged"]]))
                .std(axis=1)
                .mean(),
            }
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def all(metrics: Dict[str, np.ndarray]):
        return {
            k: {
                "mean": metrics[k].mean(),
                "std": metrics[k].std(),
            }
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def all_iq(metrics: Dict[str, np.ndarray]):
        return {
            k: (
                iq(metrics[k]).mean(),
                iq(metrics[k]).std(),
            )
            for k in metrics
            if k != "converged"
        }


class Metrics:
    @staticmethod
    def converged(rewards_df: pd.DataFrame, n_tasks: int = 2):
        return all([rewards_df.iloc[0][f"converged_{i}"] for i in range(n_tasks)])

    @staticmethod
    def final_reward(rewards_df: pd.DataFrame):
        return rewards_df["rewards"].iloc[-1]

    @staticmethod
    def transfer_area_under_curve(rewards_df: pd.DataFrame):
        novelty_step = rewards_df["novelty_step"].iloc[0]
        n_tasks = rewards_df["n_tasks"].iloc[0]
        assert n_tasks == 2
        task_one_rewards = rewards_df["rewards"][rewards_df.index <= novelty_step]
        task_two_rewards = rewards_df["rewards"][rewards_df.index > novelty_step]

        task_one_final_performance = task_one_rewards.iloc[-10:].mean()

        # task_two_range = (
        #     rewards_df.index[rewards_df.index > novelty_step].min()
        #     - rewards_df.index[rewards_df.index > novelty_step].max()
        # )

        """
        This is the normalized area under the curve since
        area = (mean value) * (range)
        full rectangle area = (reward cap) * (range) = (range)
        normalized_area = (area) / (full rectangle area) = (area) / (range) = (mean value)
        """
        task_two_area_under_the_curve = task_two_rewards.mean()

        return task_one_final_performance + task_two_area_under_the_curve


def main(args):
    df = load_data(args=args)
    metrics = calculate_metrics(
        args=args,
        df=df,
        metrics_fns={
            k: v.__func__
            for k, v in Metrics.__dict__.items()
            if isinstance(v, staticmethod)
        },
    )
    results = aggregate_metrics(
        metrics=metrics,
        aggs={
            k: v.__func__
            for k, v in Aggregators.__dict__.items()
            if isinstance(v, staticmethod)
        },
    )

    import pdb

    pdb.set_trace()

    with open("./data/metrics.pkl", "wb") as f:
        pkl.dump(metrics, file=f)
    with open("./data/results.json", "w") as f:
        json.dump(results, fp=f, indent=4)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser)
    main(args=args)
