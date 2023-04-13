import os
import sys
import numpy as np

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import gym
from stable_baselines3 import SAC

import time

from rlexplore.poly_rl.poly_rl_policy import PolyRLSACPolicy, SACPolicy

LOAD_MODEL = False
TIMESTEPS = 2000000
# ENV_NAME = "MountainCarContinuous-v0"
ENV_NAME = "Ant-v2"


def get_args():
    str2bool = lambda s: s.lower() in ["true", "1", "t", "y", "yes", "yeah"]
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--env-id", type=str, default=ENV_NAME)
    parser.add_argument("--total-time-steps", type=int, default=TIMESTEPS)
    parser.add_argument("--n-steps", type=int, default=128)

    parser.add_argument("--load-model", type=str2bool, default=LOAD_MODEL)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--poly-rl", type=str2bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    timestamp = int(time.time())

    model_name = f"{args.env_id.lower().replace('-', '_')}{'_poly_rl' if args.poly_rl else ''}_sac_{args.total_time_steps}_{timestamp}"
    model_file_path = f"models/{model_name}"
    log_path = f"logs/{model_name}"

    env = gym.make(args.env_id)

    num_episodes = int(args.total_time_steps / args.n_steps)

    n_actions = env.action_space.shape[-1]

    model = SAC(
        policy=PolyRLSACPolicy if args.poly_rl else SACPolicy,
        env=env,
        verbose=1,
        learning_rate=0.01,
        batch_size=256,
        gamma=args.gamma,
        tensorboard_log="./logs/",
        policy_kwargs=dict(logdir=log_path) if args.poly_rl else dict(),
    )

    def test_model():
        total_reward = 0
        eps = 1
        num_steps = 1000

        observation = env.reset()
        for _ in range(num_steps):
            action, _ = model.predict(observation, deterministic=True)
            action = np.array(action)
            if action.shape == ():
                action = action[np.newaxis]
            # print(action.shape, action)
            # sys.exit()
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                observation = env.reset()
                eps += 1

        print("-------------------------------------------")
        print(f"Average Rewards: {total_reward / eps}")
        print(f"Average Episode Length: {num_steps / eps}")
        print(f"Total Reward: {total_reward}")
        print(f"Num Episodes: {eps}")
        print("-------------------------------------------")

    test_model()

    if args.load_model:
        model.load(model_file_path)
    else:
        model.learn(
            total_timesteps=args.total_time_steps,
            log_interval=1,
            tb_log_name=model_name,
        )

    test_model()

    if not args.load_model:
        model.save(model_file_path)

    env.close()
