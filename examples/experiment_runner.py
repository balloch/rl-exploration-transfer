import os
import sys
from typing import Any, Dict, Optional, List, Union, Type

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import rlexplore
import stable_baselines3 as sb3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from novelty_env import NoveltyEnv
import gymnasium as gym
import torch
import time
import inspect


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
    log: bool = True,
    n_runs: int = 1,
    log_interval: int = 1,
    device: Optional[torch.device] = None,
    override_timestamp: Optional[int] = None,
    print_novelty_box: bool = False,
):
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
            k: v
            for k, v in inspect.getmembers(
                rlexplore,
                lambda obj: inspect.isclass(obj) and issubclass(obj, BasePolicy),
            )
        }.get(policy, policy)

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

        str_replacement_params = {
            "env": env,
            "envs": env,
            "torch_device": device,
            "env_observation_shape": env.observation_space.shape,
            "env_action_shape": env.action_space.shape,
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
        )

        if save_model:
            model.save(model_file_path)


default_config = {
    k: v.default
    for k, v in inspect.signature(run_experiment).parameters.items()
    if v.default is not inspect.Parameter.empty
}

if __name__ == "__main__":
    run_experiment()
