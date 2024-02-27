import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", rc={"figure.figsize": (30, 15)})


from utils.arg_types import tup
from utils.args import get_args
from experiments.wandb_run_data import make_data_loader_parser, load_data


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
    parser = make_data_loader_parser()

    parser.description = (
        "A python script to pull the reward data from wandb and plot it using seaborn."
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


def visualize_data(args: argparse.Namespace, df: pd.DataFrame) -> None:
    plot = sns.lineplot(
        x="global_step",
        y="rollout/ep_rew_mean",
        hue="experiment_name",
        data=df,
        errorbar=make_error_bar_arg(args.error_bar_type, 0),
        estimator=args.estimator,
    )
    plot.figure.savefig(f"figures/{args.img_name}")
    plt.close()

    for experiment_name_idx in set(df.index.get_level_values("experiment_name_idx")):
        plot = sns.lineplot(
            x="global_step",
            y="rollout/ep_rew_mean",
            hue="experiment_name",
            data=df.loc[experiment_name_idx],
            errorbar=make_error_bar_arg(args.error_bar_type, args.error_bar_arg),
            estimator=args.estimator,
        )
        plot.figure.savefig(f"figures/{experiment_name_idx}_{args.img_name}")
        plt.close()


def main(args):
    df = load_data(args)
    visualize_data(args, df)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser, configs_root="./configs")
    main(args)
