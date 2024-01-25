from typing import Dict, List, Any, Sequence, Union, Callable

import argparse
import gymnasium as gym
import minigrid
import json

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

ENV_NAME = "MiniGrid-Empty-8x8-v0"
CONFIG_FILE = "sample.json"
TIMESTEPS = None
NOVELTY_STEP = 10
NUM_ENVS = 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", "-e", type=str, default=ENV_NAME)
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default=CONFIG_FILE,
        help="Use the path to a json file here.",
    )
    parser.add_argument("--total-steps", "-s", type=int, default=TIMESTEPS)
    parser.add_argument("--novelty-step", "-n", type=int, default=NOVELTY_STEP)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)

    return parser


def make_env_list(
    env_name: str,
    env_configs: List[Dict[str, Any]],
    num_envs: int = 1,
    wrappers: Union[Sequence[Callable[[gym.Env], gym.Wrapper]], None] = None,
) -> List[gym.Env]:
    env_list = []

    for config in env_configs:
        if "env_id" in config:
            env_id = config["env_id"]
            config = {k: config[k] for k in config if k != "env_name"}
        envs = make_vec_env(
            env_id=env_id,
            n_envs=num_envs,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=wrappers[0],
            env_kwargs=config,
        )  # TODO: currently only supports one wrapper

        env_list.append(envs)

    return env_list


def run(
    args: argparse.Namespace,
    step_callback: Callable[[gym.Env], None] = None,
    wrappers: Union[Sequence[Callable[[gym.Env], gym.Wrapper]], bool] = None,
) -> None:
    with open(args.config_file) as f:
        env_configs = json.load(f)
    env_list = make_env_list(
        env_name=args.env_name,
        env_configs=env_configs,
        num_envs=args.num_envs,
        wrappers=wrappers,
    )

    if args.total_steps is None:
        total_steps = len(env_list) * args.novelty_step
    else:
        total_steps = args.total_steps

    env_idx = -1
    env = None

    for step_num in range(total_steps):
        if step_num % args.novelty_step == 0:
            if env_idx + 1 < len(env_list):
                if env is not None:
                    env.close()
                env_idx += 1
                env = env_list[env_idx]
                env.reset()

        if step_callback is None:
            obs, reward, truncated, terminated, info = env.step(
                env.action_space.sample()
            )
            print(
                f"env_idx: {env_idx}; step_num: {step_num}; grid_size: {env.envs[0].width}"
            )
        else:
            continue_running = step_callback(env)
            if not continue_running:
                break

    if env is not None:
        env.close()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    run(args=args)
