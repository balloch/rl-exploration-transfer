import os
import sys
from typing import Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import gymnasium as gym
import minigrid

from experiments.experiment_runner import run_experiment
from experiments.config import make_parser

from utils.args import get_args
from utils import default


def main(args):

    run_experiment(
        experiment_name=default(
            args.experiment_name,
            (
                f"{args.experiment_prefix}"
                f"{args.env_configs_file.split('.')[0]}_"
                f"{(args.rl_alg if type(args.rl_alg) == str else args.rl_alg.__name__).lower()}"
                f"{args.experiment_suffix}"
            ),
        ),
        env_configs=args.env_configs_file,
        total_time_steps=args.total_time_steps,
        novelty_step=args.novelty_step,
        n_envs=args.n_envs,
        wrappers=args.wrappers,
        model_cls=args.rl_alg,
        model_kwargs=dict(
            verbose=args.verbose,
            **args.rl_alg_kwargs,
        ),
        policy=args.policy,
        policy_kwargs=args.policy_kwargs,
        n_runs=args.n_runs,
        log=args.log,
        log_interval=args.log_interval,
        wandb_project_name=args.wandb_project_name,
        wandb_save_videos=args.wandb_save_videos,
        wandb_video_freq=args.wandb_video_freq,
        wandb_video_length=args.wandb_video_length,
        wandb_model_save_freq=args.wandb_model_save_freq,
        wandb_gradient_save_freq=args.wandb_gradient_save_freq,
        wandb_verbose=args.wandb_verbose,
        device=args.device,
        gpu_idx=args.gpu_idx,
        print_novelty_box=args.print_novelty_box,
        save_model=args.save_model,
        full_config=vars(args),
    )


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser, configs_root="./configs")
    main(args=args)
