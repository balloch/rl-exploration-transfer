import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import gym
import gym_minigrid
from stable_baselines3 import PPO
import time
# to do: add after implementation 
from rlexplore.vime.vime_policy import VimeActorCriticCnnPolicy


LOAD_MODEL = False
TIMESTEPS = 3000000
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
VIME = False

def get_args():
    str2bool = lambda s: s.lower() in ["true", "1", "t", "y", "yes", "yeah"]

    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument("--load-model", type=str2bool, default=LOAD_MODEL)
    parser.add_argument("--total-time-steps", type=int, default=TIMESTEPS)
    parser.add_argument("--env-id", type=str, default=ENV_NAME)
    parser.add_argument("--vime", type=bool, default= VIME)


    args = parser.parse_args()
    return args


class StableBaselinesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


if __name__ == "__main__":
    args = get_args()

    timestamp = int(time.time())

    model_name = f"{args.env_id.lower().replace('-', '_')}{'_vime' if args.vime else ''}_cnn_ppo_{args.total_time_steps}_{timestamp}"
    model_file_path = f"models/{model_name}"
    log_path = f"logs/{model_name}"

    env = gym.make(args.env_id)

    env = gym_minigrid.wrappers.RGBImgObsWrapper(env, tile_size=8)

    env = StableBaselinesWrapper(env)

    model = PPO(
        policy=VimeActorCriticCnnPolicy,
        env=env,
        verbose=1,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0 if args.noisy_layers else 0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            num_noisy_layers=args.num_noisy_layers if args.noisy_layers else 0
        ),
    )

    def test_model():
        total_reward = 0
        eps = 1
        num_steps = 1000
 
        observation = env.reset()
        for _ in range(num_steps):
            action, _ = model.predict(observation, deterministic=True)
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