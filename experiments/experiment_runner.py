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
import stable_baselines3.common.callbacks as sb3_callbacks
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
import torch
import time
import json
import inspect

import rlexplore
import minigrid
from novgrid import NoveltyEnv
from novgrid.env_configs import get_env_configs
import env_converters.rwrl_to_gym

from utils import get_all_subclasses_from_modules


def run_experiment(
    experiment_name: str = "experiment",
    env_configs: Union[str, Dict[str, Any]] = "sample",
    total_time_steps: int = 1_000_000,
    novelty_step: int = 250_000,
    n_envs: int = 1,
    wrappers: Optional[List[gym.Wrapper]] = None,
    wrapper_kwargs_lst: Optional[List[Optional[Dict[str, Any]]]] = None,
    model_cls: Union[str, Type[BaseAlgorithm]] = "PPO",
    model_kwargs: Optional[Dict[str, Any]] = None,
    policy: Union[str, BasePolicy] = "MlpPolicy",
    policy_kwargs: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[BaseCallback]] = None,
    callback_kwargs_lst: Optional[List[Optional[Dict[str, Any]]]] = None,
    save_model: bool = True,
    n_runs: int = 1,
    log: bool = True,
    use_wandb: bool = True,
    log_interval: int = 1,
    wandb_project_name: str = "rl-transfer-explore",
    wandb_save_videos: bool = False,
    wandb_video_freq: int = 2_000,
    wandb_video_length: int = 200,
    wandb_model_save_freq: int = 100_000,
    wandb_gradient_save_freq: int = 0,
    wandb_verbose: int = 2,
    device: str = "cuda:0",
    gpu_idx: Optional[int] = None,
    override_timestamp: Optional[int] = None,
    print_novelty_box: bool = False,
    seed: int = None,
    full_config: Dict[str, Any] = None,
):
    config = dict(locals().copy())

    if wrappers is None:
        wrappers = []
    if wrapper_kwargs_lst is None:
        wrapper_kwargs_lst = []

    if callbacks is None:
        callbacks = []
    if callback_kwargs_lst is None:
        callback_kwargs_lst = []

    wrapper_kwargs_lst = [
        ({} if wrapper_kwargs is None else wrapper_kwargs)
        for wrapper_kwargs in wrapper_kwargs_lst
    ]
    for i in range(len(wrapper_kwargs_lst), len(wrappers)):
        wrapper_kwargs_lst.append({})

    callback_kwargs_lst = [
        ({} if callback_kwargs is None else callback_kwargs)
        for callback_kwargs in callback_kwargs_lst
    ]
    for i in range(len(callback_kwargs_lst), len(callbacks)):
        callback_kwargs_lst.append({})

    if gpu_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

    if type(config["env_configs"]) == str:
        if os.path.exists(env_configs):
            with open(env_configs, "r") as f:
                config["env_configs"] = json.load(f)
        else:
            config["env_configs"] = get_env_configs(env_configs)

    device = torch.device(device)
    timestamp = int(time.time()) if override_timestamp is None else override_timestamp

    config["timestamp"] = timestamp
    config["experiment_id"] = f"{experiment_name}_{timestamp}"

    if type(model_cls) == str:
        model_cls = get_all_subclasses_from_modules(
            rlexplore, sb3, super_cls=BaseAlgorithm, lower_case_keys=True
        )[model_cls.lower()]
    if type(policy) == str:
        policy = get_all_subclasses_from_modules(
            rlexplore, sb3_policies, super_cls=BasePolicy
        ).get(policy.lower(), policy)

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

    callback_classes = get_all_subclasses_from_modules(
        rlexplore,
        sb3_callbacks,
        super_cls=BaseCallback,
        lower_case_keys=True,
    )
    for i in range(len(callbacks)):
        callbacks[i] = (
            callback_classes[callbacks[i].lower()]
            if type(callbacks[i]) == str
            else callbacks[i]
        )

    for run_num in range(n_runs):
        model_name = f"{experiment_name}_{timestamp}_{run_num}_{n_runs}"
        model_file_path = f"models/{model_name}"

        env = NoveltyEnv(
            env_configs=env_configs,
            novelty_step=novelty_step,
            n_envs=n_envs,
            wrappers=wrappers,
            wrapper_kwargs_lst=wrapper_kwargs_lst,
            print_novelty_box=print_novelty_box,
        )

        config["n_tasks"] = env.n_tasks

        if log and use_wandb:
            os.makedirs(model_file_path, exist_ok=True)
            wandb_run = wandb.init(
                id=model_name,
                project=wandb_project_name,
                config=config,
                sync_tensorboard=True,
                monitor_gym=wandb_save_videos,
                save_code=True,
                reinit=True,
                tags=[str(timestamp), experiment_name],
            )

        if wandb_save_videos and use_wandb:  # TODO: This is still not working!
            env = VecVideoRecorder(
                env,
                f"videos/{wandb_run.id}",
                record_video_trigger=lambda x: x % wandb_video_freq == 0,
                video_length=wandb_video_length,
            )
        if seed is not None:
            env.seed(seed + 2 * run_num + 1)

        env.reset()

        str_replacement_params = {
            "env": env,
            "envs": env,
            "torch_device": device,
            "env_observation_shape": env.observation_space.shape,
            "env_action_shape": env.action_space.shape,
            "none": None,
            **get_all_subclasses_from_modules(
                sb3, super_cls=BaseAlgorithm, lower_case_keys=True
            ),
            **get_all_subclasses_from_modules(rlexplore, lower_case_keys=True),
        }

        def replace_all_env_based_params(d):
            if type(d) != dict:
                return
            for k in d:
                if type(d[k]) == dict:
                    replace_all_env_based_params(d[k])
                elif type(d[k]) == str and d[k].lower() in str_replacement_params:
                    d[k] = str_replacement_params[d[k].lower()]

        if model_kwargs is None:
            model_kwargs = {}

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

        if seed is not None:
            model.set_random_seed(seed=seed + run_num)

        callback_instances = []

        if log and use_wandb:
            callback_instances.append(
                WandbCallback(
                    gradient_save_freq=wandb_gradient_save_freq,
                    model_save_freq=wandb_model_save_freq,
                    model_save_path=model_file_path,
                    verbose=wandb_verbose,
                )
            )

        def update_callback_kwargs(callback_kwargs):
            return {
                k: (
                    v
                    if type(v) != dict or "callback_cls" not in v
                    else (
                        callback_classes[v["callback_cls"].lower()]
                        if type(v["callback_cls"]) == str
                        else v["callback_cls"]
                    )(update_callback_kwargs(v.get("callback_kwargs", {})))
                )
                for k, v in callback_kwargs.items()
            }

        for callback_cls, callback_kwargs in zip(callbacks, callback_kwargs_lst):
            callback_instances.append(
                callback_cls(**update_callback_kwargs(callback_kwargs=callback_kwargs)),
            )

        model.learn(
            total_timesteps=total_time_steps,
            log_interval=log_interval,
            tb_log_name=model_name,
            callback=(
                CallbackList(callback_instances)
                if len(callback_instances) > 0
                else None
            ),
        )

        if save_model and (not use_wandb or not log):
            os.makedirs(model_file_path, exist_ok=True)
            model.save(f"{model_file_path}/final.zip")


default_config = {
    k: v.default
    for k, v in inspect.signature(run_experiment).parameters.items()
    if v.default is not inspect.Parameter.empty
}

if __name__ == "__main__":
    run_experiment()
