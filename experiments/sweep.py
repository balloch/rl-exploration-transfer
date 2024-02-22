import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import time
import numpy as np
import wandb
import gymnasium as gym
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
import minigrid
import rlexplore
from novgrid import NoveltyEnv

from experiments.experiment_runner import run_experiment
import experiments.config as experiment_config

from utils.args import get_args, remove_argument
from utils.arg_types import json_type
from utils import get_all_subclasses_from_modules

experiment_config.WANDB_PROJECT_NAME = "rl-transfer-explore-sweeps"
experiment_config.RL_ALG_KWARGS = {"learning_rate": "$learning_rate"}

SWEEP_ENV_CONFIGS = []
SWEEP_CONFIG = {}
SWEEP_COUNT = 10

EVAL_FREQ = 10000
EVAL_EPISODES = 10
MIN_REWARD_THRESHOLD = 0.75


def make_parser():
    parser = experiment_config.make_parser()

    remove_argument(parser=parser, arg="env_config_file")
    remove_argument(parser=parser, arg="novelty_step")
    remove_argument(parser=parser, arg="render_display")
    remove_argument(parser=parser, arg="step_delay")
    remove_argument(parser=parser, arg="callbacks")
    remove_argument(parser=parser, arg="callbacks_kwargs")
    remove_argument(parser=parser, arg="wandb_save_videos")
    remove_argument(parser=parser, arg="wandb_video_freq")
    remove_argument(parser=parser, arg="wandb_video_length")
    remove_argument(parser=parser, arg="wandb_model_save_freq")
    remove_argument(parser=parser, arg="wandb_gradient_save_freq")
    remove_argument(parser=parser, arg="wandb_verbose")
    remove_argument(parser=parser, arg="log")
    remove_argument(parser=parser, arg="save_model")

    parser.description = "A hyperparameter tuning script that uses wandb sweeps to tune the specified hyperparameters against the convergence speed metric."

    parser.add_argument(
        "--sweep-env-configs",
        "-secf",
        type=json_type,
        default=SWEEP_ENV_CONFIGS,
        help="Use the path to a json file containing the env configs here.",
    )
    parser.add_argument(
        "--sweep-configuration",
        "-sc",
        type=json_type,
        default=SWEEP_CONFIG,
        help="The sweep configuration to use in wandb.",
    )

    parser.add_argument(
        "--eval-freq",
        "-ef",
        type=int,
        default=EVAL_FREQ,
        help="The frequency to evaluate the agent.",
    )
    parser.add_argument(
        "--eval-episodes",
        "-ee",
        type=int,
        default=EVAL_EPISODES,
        help="The number of eval episodes to use.",
    )
    parser.add_argument(
        "--min-reward-threshold",
        "-mrt",
        type=float,
        default=MIN_REWARD_THRESHOLD,
        help="The min reward threshold to stop training in the eval callback.",
    )

    return parser


def main(args):

    wrappers = args.wrappers
    wrapper_kwargs_lst = args.wrappers_kwargs

    if wrappers is None:
        wrappers = []
    if wrapper_kwargs_lst is None:
        wrapper_kwargs_lst = []

    wrapper_kwargs_lst = [
        ({} if wrapper_kwargs is None else wrapper_kwargs)
        for wrapper_kwargs in wrapper_kwargs_lst
    ]
    for i in range(len(wrapper_kwargs_lst), len(wrappers)):
        wrapper_kwargs_lst.append({})

    wrapper_classes = get_all_subclasses_from_modules(
        rlexplore,
        minigrid.wrappers,
        gym.wrappers,
        super_cls=gym.Wrapper,
        lower_case_keys=True,
    )
    for i in range(len(wrappers)):
        wrappers[i] = (
            wrapper_classes[wrappers[i].lower()]
            if type(wrappers[i]) == str
            else wrappers[i]
        )

    timestamp = int(time.time())

    experiment_name = (
        f"{args.experiment_prefix}"
        f"{args.experiment_name}_"
        f"{(args.rl_alg if type(args.rl_alg) == str else args.rl_alg.__name__).lower()}"
        f"{args.experiment_suffix}"
    )

    def sweep_single_run():
        wandb.init(
            sync_tensorboard=False,
            save_code=True,
            reinit=True,
            tags=[str(timestamp), experiment_name, "sweep"],
        )

        config = wandb.config

        def replace_kwargs_with_config(kwargs):
            return {
                k: (
                    replace_kwargs_with_config(v)
                    if type(v) == dict
                    else (config[v[1:]] if type(v) == str and v.startswith("$") else v)
                )
                for k, v in kwargs.items()
            }

        rl_alg_kwargs = replace_kwargs_with_config(args.rl_alg_kwargs)
        policy_kwargs = replace_kwargs_with_config(args.policy_kwargs)
        wrappers_kwargs = [
            replace_kwargs_with_config(wrapper_kwargs)
            for wrapper_kwargs in args.wrappers_kwargs
        ]

        final_timesteps = []

        class MetricsCallback(BaseCallback):

            def __init__(self, verbose: int = 0):
                super().__init__(verbose)

            def _on_training_end(self) -> None:
                final_timesteps.append(self.model.num_timesteps)
                return super()._on_training_end()

            def _on_step(self) -> bool:
                return super()._on_step()

        for env_config in args.sweep_env_configs:
            eval_env = NoveltyEnv(
                env_configs=[env_config],
                novelty_step=1e10,
                wrappers=wrappers,
                wrapper_kwargs_lst=wrapper_kwargs_lst,
                n_envs=1,
            )

            eval_callback_kwargs = {
                "eval_env": eval_env,
                "eval_freq": args.eval_freq,
                "callback_on_new_best": StopTrainingOnRewardThreshold(
                    reward_threshold=args.min_reward_threshold, verbose=args.verbose
                ),
                "verbose": args.verbose,
            }

            run_experiment(
                experiment_name=experiment_name,
                env_configs=[env_config],
                total_time_steps=args.total_time_steps,
                novelty_step=1e10,
                n_envs=args.n_envs,
                wrappers=args.wrappers,
                wrapper_kwargs_lst=wrappers_kwargs,
                callbacks=[EvalCallback, MetricsCallback],
                callback_kwargs_lst=[eval_callback_kwargs, {"verbose": args.verbose}],
                model_cls=args.rl_alg,
                model_kwargs=dict(
                    verbose=args.verbose,
                    **rl_alg_kwargs,
                ),
                policy=args.policy,
                policy_kwargs=policy_kwargs,
                n_runs=args.n_runs,
                log=False,
                log_interval=args.log_interval,
                wandb_project_name=args.wandb_project_name,
                device=args.device,
                gpu_idx=args.gpu_idx,
                print_novelty_box=args.print_novelty_box,
                save_model=False,
            )

        if len(final_timesteps) > 0:
            wandb.log({"average_final_timesteps": np.mean(final_timesteps)})
        else:
            wandb.log({"average_final_timesteps": 0})

    sweep_config = {
        "name": f"{experiment_name}_sweep_{timestamp}",
        "metric": {"goal": "minimize", "name": "average_final_timesteps"},
        **args.sweep_configuration,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project_name)

    wandb.agent(
        sweep_id,
        function=sweep_single_run,
        project=args.wandb_project_name,
        count=args.sweep_count,
    )


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser, configs_root="./configs")
    main(args=args)
