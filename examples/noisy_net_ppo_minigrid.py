import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import gym
from stable_baselines3 import PPO
import gym_minigrid

# from rlexplore.noisy_nets.noisy_dqn import NoisyDQNCnnPolicy
# from rlexplore.noisy_nets.noisy_actor import
from rlexplore.noisy_nets.noisy_mlp import NoisyActorCriticCnnPolicy


class StableBaselinesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


LOAD_MODEL = False
NOISY = True
TIMESTEPS = 10000000
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
model_file_name = f"models/minigrid{'_noisy' if NOISY else ''}_cnn_ppo_{TIMESTEPS}"

if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    env = gym_minigrid.wrappers.RGBImgObsWrapper(env, tile_size=8)

    env = StableBaselinesWrapper(env)

    if NOISY:
        # model = DQN(
        #     NoisyDQNCnnPolicy,
        #     env,
        #     verbose=0,
        #     exploration_final_eps=0,
        #     exploration_fraction=0,
        #     exploration_initial_eps=0,
        # )
        model = PPO(
            policy=NoisyActorCriticCnnPolicy,
            env=env,
            verbose=1,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    else:
        # model = DQN("CnnPolicy", env, verbose=0)
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

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
        model.learn(total_timesteps=TIMESTEPS, log_interval=4)

    run()

    if not LOAD_MODEL:
        model.save(model_file_name)

    env.close()
