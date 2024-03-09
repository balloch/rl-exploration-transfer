import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from utils.args import get_args
import experiments.wandb_run_data as wrd
from experiments.wandb_run_data import make_data_loader_parser, load_data


wrd.PULL_FROM_WANDB = False
wrd.FILTER_UNCONVERGED_OUT = False
ESTIMATOR = "mean"
ERROR_BAR_TYPE = "ci"
ERROR_BAR_ARG = 95
ALGS = None
CROP_MARGIN = 500000

ALG_ORDER = [
    "None (PPO)",
    "NoisyNets",
    "ICM",
    "DIAYN",
    "RND",
    "NGU",
    "RIDE",
    "GIRL",
    "RE3",
    "RISE",
    "REVD",
]

CROP_END = wrd.CROP_END


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
        "--algs",
        "-a",
        type=str,
        nargs="+",
        default=ALGS,
        help="Which algorithms to plot.",
    )

    parser.add_argument(
        "--crop-margin",
        "-cm",
        type=int,
        default=CROP_MARGIN,
        help="The amount of margin before novelty to add on the cropped graph.",
    )

    return parser


def visualize_data(args: argparse.Namespace, df: pd.DataFrame) -> None:

    os.makedirs("./figures/rewards", exist_ok=True)

    df = df.loc[
        [
            args.algs is None or any((alg in name) for alg in args.algs)
            for name in df.index.get_level_values("experiment_name_idx")
        ]
    ]
    df = df.loc[df["global_step"] < CROP_END]
    df = df.loc[df["converged_0"]]
    replace_names = {"noisy": "NoisyNets", "none": "None (PPO)", "girm": "GIRL"}
    rename_columns = {
        "experiment_name": "Exploration Algorithm",
        "global_step": "Time Steps",
        "rollout/ep_rew_mean": "Episode Reward Mean",
    }

    def alg_name(s):
        for k in replace_names:
            if k in s:
                return replace_names[k]
        return s.split("_")[-1].upper()

    env_name = (
        df["experiment_name"]
        .iloc[0]
        .split("_tuned")[1]
        .split("_ir")[0]
        .split("_ppo")[0]
        .replace("_", " ")
        .title()
    )

    df["experiment_name"] = df["experiment_name"].apply(alg_name)

    df = df.rename(columns=rename_columns)

    n_tasks = df["n_tasks"].iloc[0]
    assert n_tasks == 2

    novelty_step = df["novelty_step"].iloc[0]
    left_crop = novelty_step - args.crop_margin

    img_name = ""
    img_name += "reward_mean"
    if args.algs is not None:
        img_name += "_" + "_".join(args.algs)
    img_name = "converged_0_" + img_name

    sns.set_theme(style="whitegrid", font_scale=3, rc={"figure.figsize": (30, 15)})

    # plt.figure()
    # plt.axvline(x=novelty_step, linestyle="--")

    # plot = sns.lineplot(
    #     x="Time Steps",
    #     y="Episode Reward Mean",
    #     hue="Exploration Algorithm",
    #     hue_order=ALG_ORDER,
    #     data=df,
    #     errorbar=make_error_bar_arg(args.error_bar_type, args.error_bar_arg),
    #     err_kws={"alpha": 0.05},
    #     estimator=args.estimator,
    # )
    # plt.title(f"{env_name} Episode Rewards")
    # plot.figure.savefig(f"figures/rewards/{img_name}.png")
    # plt.close()

    # plt.figure()
    # plt.axvline(x=novelty_step, linestyle="--")

    # plot = sns.lineplot(
    #     x="Time Steps",
    #     y="Episode Reward Mean",
    #     hue="Exploration Algorithm",
    #     data=df.loc[df["Time Steps"] >= left_crop],
    #     errorbar=make_error_bar_arg(args.error_bar_type, args.error_bar_arg),
    #     err_kws={"alpha": 0.05},
    #     estimator=args.estimator,
    # )
    # plt.title(f"{env_name} Episode Rewards")

    # plot.figure.savefig(f"figures/rewards/cropped_{img_name}.png")
    # plt.close()

    plt.figure()
    plt.axvline(x=novelty_step, linestyle="--")

    plot = sns.lineplot(
        x="Time Steps",
        y="Episode Reward Mean",
        hue="Exploration Algorithm",
        data=df.loc[df["Time Steps"] >= left_crop],
        errorbar=make_error_bar_arg(args.error_bar_type, args.error_bar_arg),
        err_kws={"alpha": 0.05},
        estimator=args.estimator,
    )
    plt.ylim(0.9, 1)
    plt.title(f"{env_name} Episode Rewards")

    plot.figure.savefig(f"figures/rewards/cropped_scaled_{img_name}.png")
    plt.close()


def main(args):
    df = load_data(args)
    visualize_data(args, df)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser, configs_root="./configs")
    main(args)
