import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import novgrid_loop_sample as novgrid

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.utils import safe_mean

import json
import time


novgrid.ENV_NAME = "MiniGrid-Empty-8x8-v0"
novgrid.CONFIG_FILE = "sample.json"
novgrid.TIMESTEPS = 3000000
novgrid.NOVELTY_STEP = 1000000
novgrid.NUM_ENVS = 1


class StableBaselinesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


wrappers = [minigrid.wrappers.RGBImgObsWrapper, StableBaselinesWrapper]


def make_parser():
    parser = novgrid.make_parser()
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    timestamp = int(time.time())

    model_name = f"{args.env_name.lower().replace('-', '_').replace('mini', 'nov')}_cnn_ppo_{args.total_steps}_{timestamp}"
    model_file_path = f"models/{model_name}"

    global model
    global iteration

    model = None
    iteration = 0

    def step_callback(env):
        global model
        global iteration

        total_timesteps = args.total_steps * 2048
        log_interval = 1
        tb_log_name = model_name
        callback = None
        reset_num_timesteps = True
        progress_bar = False

        if model is None:
            model = PPO(
                policy=ActorCriticCnnPolicy,
                env=env,
                verbose=1,
                learning_rate=1e-3,
                n_steps=2048,
                batch_size=256,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log="./logs/",
                policy_kwargs=dict(
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                ),
            )
            total_timesteps, callback = model._setup_learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=tb_log_name,
                progress_bar=progress_bar,
            )
            callback.on_training_start(locals(), globals())

        if env != model.env:
            model.set_env(env=env, force_reset=False)

        continue_training = model.collect_rollouts(
            env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps
        )

        if not continue_training:
            return False

        iteration += 1
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            assert model.ep_info_buffer is not None
            time_elapsed = max(
                (time.time_ns() - model.start_time) / 1e9, sys.float_info.epsilon
            )
            fps = int(
                (model.num_timesteps - model._num_timesteps_at_start) / time_elapsed
            )
            model.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(model.ep_info_buffer) > 0 and len(model.ep_info_buffer[0]) > 0:
                model.logger.record(
                    "rollout/ep_rew_mean",
                    safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]),
                )
                model.logger.record(
                    "rollout/ep_len_mean",
                    safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]),
                )
            model.logger.record("time/fps", fps)
            model.logger.record(
                "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
            )
            model.logger.record(
                "time/total_timesteps", model.num_timesteps, exclude="tensorboard"
            )
            model.logger.dump(step=model.num_timesteps)

    novgrid.run(
        args=args,
        step_callback=step_callback,
        wrappers=wrappers,
    )

    model.save(model_file_path)


def main2():
    parser = make_parser()
    args = parser.parse_args()

    timestamp = int(time.time())

    env_full_name = "novgrid_test"
    model_name = f"{env_full_name}_cnn_ppo_{args.total_steps}_{timestamp}"
    model_file_path = f"models/{model_name}"

    with open(args.config_file) as f:
        env_configs = json.load(f)
    env_list = novgrid.make_env_list(
        env_name=args.env_name,
        env_configs=env_configs,
        num_envs=args.num_envs,
        wrappers=wrappers,
    )

    model = PPO(
        policy=ActorCriticCnnPolicy,
        env=env_list[0],
        verbose=1,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )

    if args.total_steps is None:
        total_steps = len(env_list) * args.novelty_step
    else:
        total_steps = args.total_steps

    env_idx = -1
    env = None
    cur_total_steps = 0
    while cur_total_steps < total_steps:
        if env_idx + 1 < len(env_list):
            if env is not None:
                env.close()
            env_idx += 1
            env = env_list[env_idx]
            model.set_env(env)

        steps_remaining = total_steps - cur_total_steps
        if env_idx == len(env_list) - 1:
            steps = steps_remaining
        else:
            steps = min(steps_remaining, args.novelty_step)

        model.learn(
            total_timesteps=steps,
            log_interval=1,
            tb_log_name=model_name,
            reset_num_timesteps=cur_total_steps == 0,
        )
        cur_total_steps += steps

    if env is not None:
        env.close()

    model.save(model_file_path)


if __name__ == "__main__":
    main2()
