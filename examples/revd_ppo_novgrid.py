import os
import sys
from typing import Any, SupportsFloat, Tuple, Dict

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import time
import torch
import argparse
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from rlexplore.rnd import RND
from rlexplore.re3 import RE3
from rlexplore.revd import REVD
from rlexplore.utils import create_env, cleanup_log_dir

import novgrid_loop_sample as novgrid

import gymnasium as gym

# import gym
import json
import minigrid
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

novgrid.ENV_NAME = "MiniGrid-Empty-8x8-v0"
novgrid.CONFIG_FILE = "sample.json"
novgrid.TIMESTEPS = 7000000
novgrid.NOVELTY_STEP = 5000000
novgrid.NUM_ENVS = 10


class StableBaselinesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(
            self.observation_space["image"]
        )

    def observation(self, obs):
        return obs["image"].flatten()


def make_parser():
    parser = novgrid.make_parser()
    parser.add_argument("--n-steps", type=int, default=2048)
    return parser


def main(args):
    device = torch.device("cuda:0")

    num_episodes = int(args.total_steps / args.n_steps / args.num_envs)
    # Create vectorized environments.

    timestamp = int(time.time())
    wrappers = [
        StableBaselinesWrapper,
    ]

    env_full_name = "novgrid_test"
    model_name = f"{env_full_name}_flattened_ppo_{args.total_steps}_{timestamp}"
    model_file_path = f"models/{model_name}"

    with open(args.config_file) as f:
        env_configs = json.load(f)
    env_list = novgrid.make_env_list(
        env_name=args.env_name,
        env_configs=env_configs,
        num_envs=args.num_envs,
        wrappers=wrappers,
    )
    # Create PPO agent.
    model = PPO(
        # policy=ActorCriticCnnPolicy,
        policy="MlpPolicy",
        env=env_list[0],
        # verbose=1,
        # learning_rate=1e-3,
        n_steps=args.n_steps,
        # batch_size=256,
        # n_epochs=4,
        # gamma=0.99,
        # gae_lambda=0.95,
        # clip_range=0.2,
        # ent_coef=0.01,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        # tensorboard_log="./logs/",
        # policy_kwargs=dict(
        #     net_arch=dict(pi=[64, 64], vf=[64, 64]),
        # ),
    )

    use_exploration = True
    use_rnd = False
    use_revd = True
    if use_exploration:
        if use_revd:
            revd = REVD(
                 obs_shape = model.env.observation_space.shape,
                 action_shape = model.env.action_space.shape,
                 device = device,
                 latent_dim =128,
                 beta = 1e-2,
                 kappa = 1e-5)
        elif use_rnd:
            # Create RND module.
            rnd = RND(
                obs_shape=env_list[0].observation_space.shape,
                action_shape=env_list[0].action_space.shape,
                device=device,
                latent_dim=128,
                beta=1e-2,
                kappa=1e-5,
                lr=1e-3,
                batch_size=32,
            )
        else:
            re3 = RE3(
                obs_shape=model.env.observation_space.shape,
                action_shape=model.env.action_space.shape,
                device=device,
                latent_dim=128,
                beta=1e-2,
                kappa=1e-5,
            )

    # Set info buffer
    model.ep_info_buffer = deque(maxlen=10)
    _, callback = model._setup_learn(total_timesteps=args.total_steps)

    t_s = time.perf_counter()
    all_eps_rewards = list()
    eps_rewards = deque([0.0] * 10, maxlen=10)

    env_idx = 0
    last_transfer = 0

    for i in range(num_episodes):
        model.collect_rollouts(
            env=model.env,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=args.n_steps,
            callback=callback,
        )
        # Compute intrinsic rewards.
        if use_exploration:
            if use_revd:
                intrinsic_rewards = revd.compute_irs(
                    rollouts={"observations": model.rollout_buffer.observations},
                    time_steps=i * args.n_steps * args.num_envs
                )
            elif use_rnd:
                intrinsic_rewards = rnd.compute_irs(
                    rollouts={"observations": model.rollout_buffer.observations},
                    time_steps=i * args.n_steps * args.num_envs,
                )
            else:
                intrinsic_rewards = re3.compute_irs(
                    rollouts={"observations": model.rollout_buffer.observations},
                    time_steps=i * args.n_steps * args.num_envs,
                    k=3,
                )
            model.rollout_buffer.rewards += intrinsic_rewards[:, :, 0]
        # Update policy using the currently gathered rollout buffer.
        model.train()
        t_e = time.perf_counter()

        eps_rewards.extend([ep_info["r"] for ep_info in model.ep_info_buffer])
        all_eps_rewards.append(list(eps_rewards.copy()))
        times_steps = i * args.n_steps * args.num_envs
        if (
            times_steps - last_transfer > args.novelty_step
            and env_idx < len(env_list) - 1
        ):
            model.env.close()
            env_idx += 1
            last_transfer += args.novelty_step
            print("----------------------------------------")
            print(f"| TRANSFER INJECTED: env_idx={env_idx} |")
            print("----------------------------------------")
            model.set_env(env_list[env_idx], force_reset=False)
            model.env.reset()

        print(
            "TOTAL TIME STEPS {}, FPS {} \n \
            MEAN|MEDIAN REWARDS {:.2f}|{:.2f}, MIN|MAX REWARDS {:.2f}|{:.2f}\n".format(
                times_steps,
                int(times_steps / (t_e - t_s)),
                np.mean(eps_rewards),
                np.median(eps_rewards),
                np.min(eps_rewards),
                np.max(eps_rewards),
            )
        )

    model.env.close()

    model.save(model_file_path)


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)