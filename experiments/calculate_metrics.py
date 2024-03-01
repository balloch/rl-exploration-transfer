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
import matplotlib.pyplot as plt

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

METRICS_TO_USE = None
AGGREGATORS_TO_USE = None


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

    parser.add_argument(
        "--metrics-to-use",
        "-mtu",
        type=str,
        default=METRICS_TO_USE,
        nargs="+",
        help="The metrics to calculate and generate plots for.",
    )
    parser.add_argument(
        "--aggregators-to-use",
        "-atu",
        type=str,
        default=AGGREGATORS_TO_USE,
        nargs="+",
        help="The aggregators to calculate and generate plots for.",
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


def iq(arr: np.ndarray, quartile_margin: int = 25):
    sorted_arr = np.sort(arr, axis=-1)
    n = sorted_arr.shape[-1]
    return sorted_arr[
        ...,
        int(np.ceil(quartile_margin / 100 * n)) : int(
            np.floor((1 - quartile_margin / 200) * n)
        )
        + 1,
    ]


def bootstrapped_sampling(arr: np.ndarray, k: int = 6, m: int = 5):
    n = len(arr)
    idx = np.array([np.random.choice(n, size=(k,), replace=False) for _ in range(m)])
    return arr[idx].flatten()


def calc_stats(arr: np.ndarray, ci_percentile: int = 95, prefix: str = ""):
    margin = (100 - ci_percentile) / 2
    return {
        f"{prefix}mean": arr.mean(axis=-1).mean(),
        f"{prefix}std": arr.std(axis=-1).mean(),
        f"{prefix}max": arr.max(axis=-1).mean(),
        f"{prefix}min": arr.min(axis=-1).mean(),
        f"{prefix}ci_{ci_percentile}_lower": np.percentile(arr, margin, axis=-1).mean(),
        f"{prefix}ci_{ci_percentile}_upper": np.percentile(
            arr, 100 - margin, axis=-1
        ).mean(),
        f"{prefix}arr": list(arr.reshape((-1, arr.shape[-1])).mean(0)),
    }


def calc_arr_and_iq_stats(arr: np.ndarray, ci_percentile: int = 95):
    return {
        **calc_stats(arr, ci_percentile=ci_percentile),
        **calc_stats(iq(arr), ci_percentile=ci_percentile, prefix="iq_"),
    }


class Aggregators:

    @staticmethod
    def converged(metrics: Dict[str, np.ndarray]):
        return {
            k: calc_arr_and_iq_stats(metrics[k][metrics["converged"]])
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def bootstrapped_converged(metrics: Dict[str, np.ndarray]):
        return {
            k: calc_arr_and_iq_stats(
                bootstrapped_sampling(metrics[k][metrics["converged"]])
            )
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def all(metrics: Dict[str, np.ndarray]):
        return {
            k: calc_arr_and_iq_stats(metrics[k]) for k in metrics if k != "converged"
        }


class Metrics:
    @staticmethod
    def converged(rewards_df: pd.DataFrame, n_tasks: int = 2):
        return all([rewards_df.iloc[0][f"converged_{i}"] for i in range(n_tasks)])

    @staticmethod
    def final_reward(rewards_df: pd.DataFrame):
        return rewards_df["rewards"].iloc[-10:].mean()

    @staticmethod
    def resiliance(rewards_df: pd.DataFrame):
        novelty_step = rewards_df["novelty_step"].iloc[0]
        return rewards_df["rewards"][rewards_df.index > novelty_step].min()

    @staticmethod
    def adaptive_efficiency(rewards_df: pd.DataFrame):
        return Metrics.percentile_efficiency(rewards_df=rewards_df, percetile=99)

    @staticmethod
    def k_shot_efficiency(rewards_df: pd.DataFrame, k: int = 100):
        novelty_step = rewards_df["novelty_step"].iloc[0]
        task_two_rewards = rewards_df["rewards"][rewards_df.index > novelty_step]
        return task_two_rewards.rolling(window=5, center=True).mean().iloc[k]

    @staticmethod
    def percentile_efficiency(rewards_df: pd.DataFrame, percetile: int = 50):
        novelty_step = rewards_df["novelty_step"].iloc[0]
        final_reward = Metrics.final_reward(rewards_df=rewards_df)
        task_two_rewards = rewards_df["rewards"][rewards_df.index > novelty_step]
        return (
            task_two_rewards[
                task_two_rewards.rolling(window=5, center=True).mean()
                >= percetile / 100 * final_reward
            ].index.min()
            - task_two_rewards.index.min()
        )

    @staticmethod
    def transfer_area_under_curve(rewards_df: pd.DataFrame):
        novelty_step = rewards_df["novelty_step"].iloc[0]
        n_tasks = rewards_df["n_tasks"].iloc[0]
        assert n_tasks == 2, "Only two tasks allowed for now"
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

        return (task_one_final_performance + task_two_area_under_the_curve) / 2


def plot_results(results: Dict[str, Dict[str, Any]]):
    labels = list(results.keys())
    plots_to_generate = list(list(results.values())[0].keys())
    metric_names = list(list(list(results.values())[0].values())[0].keys())
    height = 0.6

    data = [
        [
            [results[k][plot_name][metric] for k in labels]
            for plot_name in plots_to_generate
        ]
        for metric in metric_names
    ]

    for i in range(len(metric_names)):
        metric = metric_names[i]
        os.makedirs(f"./figures/{metric}", exist_ok=True)
        for j in range(len(plots_to_generate)):
            plot_name = plots_to_generate[j]
            plot_data = data[i][j]
            for prefix in ["", "iq_"]:
                plt.figure(figsize=(30, 15))
                ax = plt.gca()
                for k in range(len(plot_data)):
                    d = plot_data[k]
                    ax.barh(
                        y=k,
                        left=d[f"{prefix}ci_95_lower"],
                        width=d[f"{prefix}ci_95_upper"] - d[f"{prefix}ci_95_lower"],
                        alpha=0.7,
                        # color="red",
                        label=labels[k].split("_")[-1],
                        height=height,
                    )
                    ax.vlines(
                        x=d[f"{prefix}mean"],
                        ymin=k - height / 2,
                        ymax=k + height / 2,
                        label=labels[k].split("_")[-1],
                        color="k",
                        alpha=1,
                    )
                ax.set_yticks(list(range(len(labels))))
                ax.set_yticklabels(
                    [label.split("_")[-1] for label in labels], fontsize="xx-large"
                )
                # plt.boxplot(plot_data, labels=labels, vert=False, showfliers=False)
                plt.title(f"{metric}_{plot_name}")
                plt.savefig(f"./figures/{metric}/{prefix}{plot_name}.png")
                plt.close()


def main(args):
    metrics_to_use = (
        Metrics.__dict__.keys()
        if args.metrics_to_use is None
        else set([k.lower() for k in args.metrics_to_use])
    )
    aggregators_to_use = (
        Aggregators.__dict__.keys()
        if args.aggregators_to_use is None
        else set([k.lower() for k in args.aggregators_to_use])
    )
    df = load_data(args=args)
    metrics = calculate_metrics(
        args=args,
        df=df,
        metrics_fns={
            k: v.__func__
            for k, v in Metrics.__dict__.items()
            if isinstance(v, staticmethod) and k.lower() in metrics_to_use
        },
    )
    results = aggregate_metrics(
        metrics=metrics,
        aggs={
            k: v.__func__
            for k, v in Aggregators.__dict__.items()
            if isinstance(v, staticmethod) and k.lower() in aggregators_to_use
        },
    )
    plot_results(results=results)

    with open("./data/metrics.pkl", "wb") as f:
        pkl.dump(metrics, file=f)
    with open("./data/results.json", "w") as f:
        json.dump(results, fp=f, indent=4)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser)
    main(args=args)
