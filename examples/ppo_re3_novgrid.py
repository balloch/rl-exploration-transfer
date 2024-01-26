import os
import sys
from typing import Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import gymnasium as gym
import torch
import time

import novelty_env as novgrid

from rlexplore.ir_model import IR_PPO
from rlexplore.re3 import RE3

novgrid.CONFIG_FILE = "sample2.json"
novgrid.TOTAL_TIME_STEPS = 1000000
novgrid.NOVELTY_STEP = 250000
novgrid.N_ENVS = 1
LOG = True
SAVE_MODEL = True


class ImageWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(
            self.observation_space["image"]
        )

    def observation(self, obs: Any) -> Any:
        return obs["image"].flatten()


def make_parser() -> argparse.ArgumentParser:
    parser = novgrid.make_parser()

    def str2bool(s: str) -> bool:
        return s.lower() in {"true", "t", "yes", "y"}

    parser.add_argument("--log", "-l", type=str2bool, default=LOG)
    parser.add_argument("--save-model", "-s", type=str2bool, default=SAVE_MODEL)

    return parser


def main(args: argparse) -> None:
    device = torch.device("cuda:0")
    wrappers = [ImageWrapper]

    timestamp = int(time.time())

    env_full_name = "novgrid_empty"
    exploration_name = "re3"
    rl_alg_name = "ppo"
    model_name = f"{env_full_name}_{rl_alg_name}_{exploration_name}_{args.total_time_steps}_{timestamp}"
    model_file_path = f"models/{model_name}"

    env = novgrid.NoveltyEnv(
        env_configs=args.config_file,
        novelty_step=args.novelty_step,
        n_envs=args.n_envs,
        wrappers=wrappers,
        print_novelty_box=True,
    )

    model = IR_PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        tensorboard_log="./logs/" if args.log else None,
        ir_alg_cls=RE3,
        ir_alg_kwargs=dict(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            latent_dim=128,
            beta=1e-2,
            kappa=1e-5,
        ),
        compute_irs_kwargs=dict(
            k=3,
        ),
    )

    model.learn(
        total_timesteps=args.total_time_steps,
        log_interval=1,
        tb_log_name=model_name,
    )

    if args.save_model:
        model.save(model_file_path)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args=args)
