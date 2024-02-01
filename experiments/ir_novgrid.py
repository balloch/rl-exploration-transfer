import os
import sys
from typing import Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import gymnasium as gym

import novelty_env as novgrid
from experiments.experiment_runner import run_experiment

from utils.arg_types import str2bool
from utils.args import get_args

novgrid.ENV_CONFIG_FILE = "simple_to_lava_to_simple_crossing.json"
novgrid.TOTAL_TIME_STEPS = 10_000_000
novgrid.NOVELTY_STEP = 3_000_000
novgrid.N_ENVS = 5

EXPERIMENT_NAME = "novgrid_simple_to_lava_to_simple_crossing_ppo_re3"

SB3_MODEL = "PPO"
POLICY = "MlpPolicy"

IR_ALG = "RE3"
IR_BETA = 1e-2
IR_KAPPA = 1e-5
IR_LR = 1e-3
IR_LATENT_DIM = 128
IR_BATCH_SIZE = 32
IR_LAMBDA = 1e-1

WANDB_PROJECT_NAME = "rl-transfer-explore"
WANDB_SAVE_VIDEOS = False
WANDB_VIDEO_FREQ = 2000
WANDB_VIDEO_LENGTH = 200
WANDB_MODEL_SAVE_FREQ = 100000
WANDB_GRADIENT_SAVE_FREQ = 0
WANDB_VERBOSE = 2

N_RUNS = 5

LOG = True
SAVE_MODEL = True

LOG_INTERVAL = 1
PRINT_NOVELTY_BOX = True
VERBOSE = 1


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

    parser.description = "An experiment runner script for intrinsic reward exploration algorithms running on environments with transfers embedding in the training."

    parser.add_argument(
        "--experiment-name",
        "-en",
        type=str,
        default=EXPERIMENT_NAME,
        help="The name of the experiment.",
    )

    parser.add_argument(
        "--sb3-model",
        "-m",
        type=str,
        default=SB3_MODEL,
        help="The name of the stable baselines model to use. Examples include PPO, DQN, etc.",
    )
    parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default=POLICY,
        help="The type of policy to use. Examples include MlpPolicy, CnnPolicy, etc.",
    )

    parser.add_argument(
        "--ir-alg",
        "-ir",
        type=str,
        default=IR_ALG,
        help="The intrinsic reward algorithm to use. Examples include RE3, RND, NGU, etc.",
    )
    parser.add_argument(
        "--ir-beta",
        "-irb",
        type=float,
        default=IR_BETA,
        help="The beta parameter for the intrinsic reward algorithm.",
    )
    parser.add_argument(
        "--ir-kappa",
        "-irk",
        type=float,
        default=IR_KAPPA,
        help="The kappa parameter for the intrinsic reward algorithm.",
    )
    parser.add_argument(
        "--ir-learning-rate",
        "-irl",
        type=float,
        default=IR_LR,
        help="The learning rate parameter for the intrinsic reward algorithm.",
    )
    parser.add_argument(
        "--ir-latent-dim",
        "-irld",
        type=int,
        default=IR_LATENT_DIM,
        help="The latent dim parameter for the intrinsic reward algorithm.",
    )
    parser.add_argument(
        "--ir-batch-size",
        "-irbs",
        type=int,
        default=IR_BATCH_SIZE,
        help="The batch size parameter for the intrinsic reward algorithm.",
    )
    parser.add_argument(
        "--ir-lambda",
        "-irlb",
        type=float,
        default=IR_LAMBDA,
        help="The lambda parameter for the intrinsic reward algorithm.",
    )

    parser.add_argument(
        "--wandb-project-name",
        "-wpn",
        type=str,
        default=WANDB_PROJECT_NAME,
        help="The project name to save under in wandb.",
    )
    parser.add_argument(
        "--wandb-save-videos",
        "-wsv",
        type=str2bool,
        default=WANDB_SAVE_VIDEOS,
        help="Whether or not to save videos to wandb.",
    )
    parser.add_argument(
        "--wandb-video-freq",
        "-wvf",
        type=int,
        default=WANDB_VIDEO_FREQ,
        help="How often to save videos to wandb.",
    )
    parser.add_argument(
        "--wandb-video-length",
        "-wvl",
        type=int,
        default=WANDB_VIDEO_LENGTH,
        help="How long the videos saved to wandb should be",
    )
    parser.add_argument(
        "--wandb-model-save-freq",
        "-wmsf",
        type=int,
        default=WANDB_MODEL_SAVE_FREQ,
        help="How often to save the model.",
    )
    parser.add_argument(
        "--wandb-gradient-save-freq",
        "-wgsf",
        type=int,
        default=WANDB_GRADIENT_SAVE_FREQ,
        help="How often to save the gradients.",
    )
    parser.add_argument(
        "--wandb-verbose",
        "-wv",
        type=int,
        default=WANDB_VERBOSE,
        help="The verbosity setting for wandb.",
    )

    parser.add_argument(
        "--n-runs", "-r", type=int, default=N_RUNS, help="The number of runs to do."
    )

    parser.add_argument(
        "--log",
        "-l",
        type=str2bool,
        default=LOG,
        help="Whether or not to log the results to tensor board and wandb.",
    )
    parser.add_argument(
        "--save-model",
        "-s",
        type=str2bool,
        default=SAVE_MODEL,
        help="Whether or not to save the model if wandb didn't already.",
    )

    parser.add_argument(
        "--log-interval",
        "-li",
        type=int,
        default=LOG_INTERVAL,
        help="The log interval for model.learn.",
    )
    parser.add_argument(
        "--print-novelty-box",
        "-pnb",
        type=str2bool,
        default=PRINT_NOVELTY_BOX,
        help="Whether or not to print the novelty box when novelty occurs.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=VERBOSE,
        help="The verbosity parameter for model.learn.",
    )

    return parser


def main(args):
    ir_alg_kwargs = dict(
        RE3=dict(
            obs_shape="env_observation_shape",
            action_shape="env_action_shape",
            device="torch_device",
            latent_dim=IR_LATENT_DIM,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        RISE=dict(
            obs_shape="env_observation_shape",
            action_shape="env_action_shape",
            device="torch_device",
            latent_dim=IR_LATENT_DIM,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        RIDE=dict(
            obs_shape="env_observation_shape",
            action_shape="env_action_shape",
            device="torch_device",
            latent_dim=args.ir_latent_dim,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        REVD=dict(
            obs_shape="env_observation_shape",
            action_shape="env_action_shape",
            device="torch_device",
            latent_dim=args.ir_latent_dim,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        NGU=dict(
            envs="envs",
            device="torch_device",
            latent_dim=args.ir_latent_dim,
            lr=args.ir_learning_rate,
            batch_size=args.ir_batch_size,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        RND=dict(
            obs_shape="env_observation_shape",
            action_shape="env_action_shape",
            device="torch_device",
            latent_dim=args.ir_latent_dim,
            lr=args.ir_learning_rate,
            batch_size=args.ir_batch_size,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        ICM=dict(
            envs="envs",
            device="torch_device",
            lr=args.ir_learning_rate,
            batch_size=args.ir_batch_size,
            beta=args.ir_beta,
            kappa=args.ir_kappa,
        ),
        GIRM=dict(
            envs="envs",
            device="torch_device",
            latent_dim=args.ir_latent_dim,
            lr=args.ir_learning_rate,
            batch_size=IR_BATCH_SIZE,
            lambd=args.ir_lambda,
            beta=IR_BETA,
            kappa=args.ir_kappa,
        ),
    )
    compute_irs_kwargs = dict(
        RE3=dict(
            k=3,
        ),
        RISE=dict(),
        RIDE=dict(),
        REVD=dict(),
        NGU=dict(),
        RND=dict(),
        ICM=dict(),
        GIRM=dict(),
    )
    model_extra_kwargs = dict(
        PPO=dict(
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
    )
    policy_kwargs = dict(
        ActorCriticCnnPolicy=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    )

    run_experiment(
        experiment_name=args.experiment_name,
        env_configs=f"env_configs/{args.env_config_file}",
        total_time_steps=args.total_time_steps,
        novelty_step=args.novelty_step,
        n_envs=args.n_envs,
        wrappers=[ImageWrapper],
        model_cls=f"IR_{args.sb3_model}",
        model_kwargs=dict(
            verbose=args.verbose,
            **model_extra_kwargs.get(args.sb3_model, {}),
            ir_alg_cls=args.ir_alg,
            ir_alg_kwargs=ir_alg_kwargs.get(args.ir_alg, {}),
            compute_irs_kwargs=compute_irs_kwargs.get(args.ir_alg, {}),
        ),
        policy=args.policy,
        policy_kwargs=policy_kwargs.get(args.policy, None),
        n_runs=args.n_runs,
        log=args.log,
        log_interval=args.log_interval,
        wandb_project_name=args.wandb_project_name,
        wandb_save_videos=args.wandb_save_videos,
        wandb_video_freq=args.wandb_video_freq,
        wandb_video_length=args.wandb_video_length,
        wandb_model_save_freq=args.wandb_model_save_freq,
        wandb_gradient_save_freq=args.wandb_gradient_save_freq,
        wandb_verbose=args.wandb_verbose,
        print_novelty_box=args.print_novelty_box,
        save_model=args.save_model,
    )


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser=parser, configs_root="./configs")
    main(args=args)
