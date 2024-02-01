import os
import sys
from typing import Any, Dict, Optional, List, Union, Type

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import stable_baselines3 as sb3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
import stable_baselines3.common.policies as sb3_policies
from stable_baselines3.common.vec_env import VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
import torch
import time
import json
import inspect

import rlexplore
from novelty_env import NoveltyEnv


def run_experiment(
    experiment_name: str = "experiment",
    env_configs: Union[str, Dict[str, Any]] = "sample2.json",
    total_time_steps: int = 1_000_000,
    novelty_step: int = 250_000,
    n_envs: int = 1,
    wrappers: List[gym.Wrapper] = [],
    wrapper_kwargs_lst: List[Dict[str, Any]] = [],
    model_cls: Union[str, Type[BaseAlgorithm]] = "PPO",
    model_kwargs: Dict[str, Any] = dict(
        verbose=1,
        n_steps=2048,
    ),
    policy: Union[str, BasePolicy] = "MlpPolicy",
    policy_kwargs: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    n_runs: int = 1,
    log: bool = True,
    use_wandb: bool = True,
    log_interval: int = 1,
    wandb_project_name: str = "rl-transfer-explore",
    wandb_save_videos: bool = False,
    wandb_video_freq: int = 2000,
    wandb_video_length: int = 200,
    wandb_model_save_freq: int = 100000,
    wandb_gradient_save_freq: int = 0,
    wandb_verbose: int = 2,
    device: Optional[torch.device] = None,
    override_timestamp: Optional[int] = None,
    print_novelty_box: bool = False,
):
    config = dict(locals().copy())

    if type(config["env_configs"]) == str:
        with open(config["env_configs"], "r") as f:
            config["env_configs"] = json.load(f)

    device = torch.device("cuda:0") if device is None else device
    timestamp = int(time.time()) if override_timestamp is None else override_timestamp

    if type(model_cls) == str:
        model_cls = {
            **{
                k: v
                for k, v in inspect.getmembers(
                    sb3,
                    lambda obj: inspect.isclass(obj) and issubclass(obj, BaseAlgorithm),
                )
            },
            **{
                k: v
                for k, v in inspect.getmembers(
                    rlexplore,
                    lambda obj: inspect.isclass(obj) and issubclass(obj, BaseAlgorithm),
                )
            },
        }[model_cls]

    if type(policy) == str:
        policy = {
            **{
                k: v
                for k, v in inspect.getmembers(
                    rlexplore,
                    lambda obj: inspect.isclass(obj) and issubclass(obj, BasePolicy),
                )
            },
            **{
                k: v
                for k, v in inspect.getmembers(
                    sb3_policies,
                    lambda obj: inspect.isclass(obj) and issubclass(obj, BasePolicy),
                )
            },
        }.get(policy, policy)

    for run_num in range(n_runs):
        model_name = f"{experiment_name}_{timestamp}_{run_num}_{n_runs}"
        model_file_path = f"models/{model_name}"
        os.makedirs(model_file_path, exist_ok=True)

        if log and use_wandb:
            wandb_run = wandb.init(
                id=model_name,
                project=wandb_project_name,
                config=config,
                sync_tensorboard=True,
                monitor_gym=wandb_save_videos,
                save_code=True,
                reinit=True,
            )

        env = NoveltyEnv(
            env_configs=env_configs,
            novelty_step=novelty_step,
            n_envs=n_envs,
            wrappers=wrappers,
            wrapper_kwargs_lst=wrapper_kwargs_lst,
            print_novelty_box=print_novelty_box,
        )
        if wandb_save_videos and use_wandb:  # TODO: This is still not working!
            env = VecVideoRecorder(
                env,
                f"videos/{wandb_run.id}",
                record_video_trigger=lambda x: x % wandb_video_freq == 0,
                video_length=wandb_video_length,
            )

        str_replacement_params = {
            "env": env,
            "envs": env,
            "torch_device": device,
            "env_observation_shape": env.observation_space.shape,
            "env_action_shape": env.action_space.shape,
            "None": None,
            **{
                k: v
                for k, v in inspect.getmembers(
                    sb3,
                    lambda obj: inspect.isclass(obj) and issubclass(obj, BaseAlgorithm),
                )
            },
            **{
                k: v
                for k, v in inspect.getmembers(
                    rlexplore,
                    lambda obj: inspect.isclass(obj),
                )
            },
        }

        def replace_all_env_based_params(d):
            if type(d) != dict:
                return
            for k in d:
                if type(d[k]) == dict:
                    replace_all_env_based_params(d[k])
                elif type(d[k]) == str and d[k] in str_replacement_params:
                    d[k] = str_replacement_params[d[k]]

        replace_all_env_based_params(policy_kwargs)
        replace_all_env_based_params(model_kwargs)

        model = model_cls(
            policy=policy,
            env=env,
            policy_kwargs=policy_kwargs,
            device=device,
            tensorboard_log="./logs/" if log else None,
            **model_kwargs,
        )

        model.learn(
            total_timesteps=total_time_steps,
            log_interval=log_interval,
            tb_log_name=model_name,
            callback=WandbCallback(
                gradient_save_freq=wandb_gradient_save_freq,
                model_save_freq=wandb_model_save_freq,
                model_save_path=model_file_path,
                verbose=wandb_verbose,
            )
            if log and use_wandb
            else None,
        )

        if save_model and (not use_wandb or not log):
            model.save(f"{model_file_path}/final.zip")


default_config = {
    k: v.default
    for k, v in inspect.signature(run_experiment).parameters.items()
    if v.default is not inspect.Parameter.empty
}

if __name__ == "__main__":
    run_experiment()
