import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import wandb
import tqdm
import pandas as pd


from config import WANDB_PROJECT_NAME
from utils.args import get_args

CONVERGENCE_REWARD_THRESHOLD = 0.8
CONVERGENCE_CHECK_STEP_RATIO = 0.9


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb-project-name",
        "-wpn",
        type=str,
        default=WANDB_PROJECT_NAME,
        help="The project name to load from in wandb.",
    )
    parser.add_argument(
        "--convergence-reward-threshold",
        "-crt",
        type=float,
        default=CONVERGENCE_REWARD_THRESHOLD,
        help="The convergence threshold on the reward.",
    )
    parser.add_argument(
        "--convergence-check-step-ratio",
        "-ccsr",
        type=float,
        default=CONVERGENCE_CHECK_STEP_RATIO,
        help="The convergence step ratio to check on each environment. If the ratio is 0.9 and the environment was trained on for 10 steps, the code will check at step 9.",
    )

    return parser


def main(args):

    api = wandb.Api()

    wandb_runs = api.runs(args.wandb_project_name, include_sweeps=False)

    for wandb_run in tqdm.tqdm(wandb_runs):
        df = pd.DataFrame(
            wandb_run.scan_history(keys=["global_step", "rollout/ep_rew_mean"])
        )
        df["global_step"] = df["global_step"].astype(int)
        df = df.set_index("global_step")["rollout/ep_rew_mean"]

        n_tasks = wandb_run.config["n_tasks"]
        novelty_step = wandb_run.config["novelty_step"]
        total_time_steps = df.index.max()

        for task_num in range(n_tasks):
            if task_num != n_tasks - 1:
                target_idx = novelty_step * (
                    task_num + args.convergence_check_step_ratio
                )
            else:
                target_idx = (
                    novelty_step * task_num * (1 - args.convergence_check_step_ratio)
                    + total_time_steps * args.convergence_check_step_ratio
                )
            idx = (df.index.to_series() - target_idx).abs().idxmin()
            convergence_tag = f"converged_{task_num}"
            wandb_run.tags = [tag for tag in wandb_run.tags if tag != convergence_tag]
            if df[idx] > args.convergence_reward_threshold:
                wandb_run.tags.append(convergence_tag)
            wandb_run.update()


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser, configs_root="./configs")
    main(args)
