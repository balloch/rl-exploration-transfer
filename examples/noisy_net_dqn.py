import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import gym
from stable_baselines3 import DQN
import numpy as np
import gym_minigrid

from rlexplore.noisy_nets.noisy_dqn import NoisyCnnPolicy


class StableBaselinesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


LOAD_MODEL = False
NOISY = True
model_file_name = "models/minigrid_noisy_cnn_dqn"

if not NOISY:
    model_file_name = "models/minigrid_cnn_dqn"

if __name__ == "__main__":
    env = gym.make("MiniGrid-DoorKey-8x8-v0")

    env = gym_minigrid.wrappers.RGBImgObsWrapper(env, tile_size=8)

    env = StableBaselinesWrapper(env)

    if NOISY:
        model = DQN(
            NoisyCnnPolicy,
            env,
            verbose=0,
            exploration_final_eps=0,
            exploration_fraction=0,
            exploration_initial_eps=0,
        )
    else:
        model = DQN("CnnPolicy", env, verbose=0)

    def run():

        total_reward = 0
        eps = 1
        num_steps = 1000

        observation = env.reset()
        for _ in range(num_steps):
            # action = env.action_space.sample()
            action, _ = model.predict(observation, deterministic=True)
            # observation, reward, terminated, truncated = env.step(action) # minigrid
            observation, reward, done, info = env.step(action)  # cart pole
            total_reward += reward
            # print(observation)

            # env.render()

            # if terminated or truncated: # minigrid
            if done:
                observation = env.reset()
                eps += 1

        print("-------------------------------------------")
        print(f"Average Rewards: {total_reward / eps}")
        print(f"Average Episode Length: {num_steps / eps}")
        print(f"Total Reward: {total_reward}")
        print(f"Num Episodes: {eps}")
        print("-------------------------------------------")

    if LOAD_MODEL:
        model.load(model_file_name)
    else:
        run()
        model.learn(total_timesteps=10000000, log_interval=4)

    run()

    if not LOAD_MODEL:
        model.save(model_file_name)

    env.close()
