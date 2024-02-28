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


class Aggregators:

    @staticmethod
    def converged_mean_std(metrics: Dict[str, np.ndarray]):
        return {
            k: (
                metrics[k][metrics["converged"]].mean(),
                metrics[k][metrics["converged"]].std(),
            )
            for k in metrics
            if k != "converged"
        }

    @staticmethod
    def all_mean_std(metrics: Dict[str, np.ndarray]):
        return {
            k: (
                metrics[k].mean(),
                metrics[k].std(),
            )
            for k in metrics
            if k != "converged"
        }

    # @staticmethod
    # def converged_iq_mean_std(metrics: Dict[str, np.ndarray]):
    #     import pdb

    #     return {
    #         k: (
    #             pdb.set_trace(),
    #             metrics[k][metrics["converged"]].mean(),
    #             metrics[k][metrics["converged"]].std(),
    #         )
    #         for k in metrics
    #         if k != "converged"
    #     }

    """
    4 agg
    all converged mean, std
    all converged iqm, iq_std

    all mean, std
    all iqm, iq_std

    (if i have time)
    bootstrapped all converged mean, std
    bootstrapped all converged iqm, iq_std
        
    """


class Metrics:
    @staticmethod
    def converged(rewards_df: pd.DataFrame, n_tasks: int = 2):
        return all([rewards_df.iloc[0][f"converged_{i}"] for i in range(n_tasks)])

    @staticmethod
    def final_reward(rewards_df: pd.DataFrame):
        return rewards_df["rewards"].iloc[-1]
    
    @staticmethod
    def transfer_area_under_curve(rewards_df: pd.DataFrame):
        return 0
    
    '''
    tr-auc (transfer area under the curve) = (sq) normalized area on second task under the curve + final task reward on first task
    '''


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
    with open("./data/results.pkl", "wb") as f:
        pkl.dump(results, file=f)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser)
    main(args=args)
