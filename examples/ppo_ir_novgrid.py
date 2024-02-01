import os
import sys
from typing import Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import gymnasium as gym
import novelty_env as novgrid
from examples.experiment_runner import run_experiment

novgrid.CONFIG_FILE = "sample3.json"
novgrid.TOTAL_TIME_STEPS = 8000000
novgrid.NOVELTY_STEP = 2000000
novgrid.N_ENVS = 5

EXPERIMENT_NAME = "novgrid_empty_ppo_re3"

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

    def str2bool(s: str) -> bool:
        return s.lower() in {"true", "t", "yes", "y"}

    parser.add_argument("--experiment-name", "-en", type=str, default=EXPERIMENT_NAME)

    parser.add_argument("--sb3-model", "-m", type=str, default=SB3_MODEL)
    parser.add_argument("--policy", "-p", type=str, default=POLICY)

    parser.add_argument("--ir-alg", "-ir", type=str, default=IR_ALG)
    parser.add_argument("--ir-beta", "-irb", type=float, default=IR_BETA)
    parser.add_argument("--ir-kappa", "-irk", type=float, default=IR_KAPPA)
    parser.add_argument("--ir-learning-rate", "-irl", type=float, default=IR_LR)
    parser.add_argument("--ir-latent-dim", "-irld", type=int, default=IR_LATENT_DIM)
    parser.add_argument("--ir-batch-size", "-irbs", type=int, default=IR_BATCH_SIZE)
    parser.add_argument("--ir-lambda", "-irlb", type=float, default=IR_LAMBDA)

    parser.add_argument(
        "--wandb-project-name", "-wpn", type=str, default=WANDB_PROJECT_NAME
    )
    parser.add_argument(
        "--wandb-save-videos", "-wsv", type=str, default=WANDB_SAVE_VIDEOS
    )
    parser.add_argument(
        "--wandb-video-freq", "-wvf", type=int, default=WANDB_VIDEO_FREQ
    )
    parser.add_argument(
        "--wandb-video-length", "-wvl", type=int, default=WANDB_VIDEO_LENGTH
    )
    parser.add_argument(
        "--wandb-model-save-freq", "-wmsf", type=int, default=WANDB_MODEL_SAVE_FREQ
    )
    parser.add_argument(
        "--wandb-gradient-save-freq",
        "-wgsf",
        type=int,
        default=WANDB_GRADIENT_SAVE_FREQ,
    )
    parser.add_argument("--wandb-verbose", "-wv", type=int, default=WANDB_VERBOSE)

    parser.add_argument("--n-runs", "-r", type=int, default=N_RUNS)

    parser.add_argument("--log", "-l", type=str2bool, default=LOG)
    parser.add_argument("--save-model", "-s", type=str2bool, default=SAVE_MODEL)

    parser.add_argument("--log-interval", "-li", type=int, default=LOG_INTERVAL)
    parser.add_argument(
        "--print-novelty-box", "-pnb", type=str2bool, default=PRINT_NOVELTY_BOX
    )
    parser.add_argument("--verbose", "-v", type=int, default=VERBOSE)

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
        env_configs=f"./configs/{args.config_file}",
        total_time_steps=args.total_time_steps,
        novelty_step=args.novelty_step,
        n_envs=args.n_envs,
        wrappers=[ImageWrapper],
        model_cls=f"IR_{args.sb3_model}",
        model_kwargs=dict(
            verbose=args.verbose,
            **model_extra_kwargs.get(args.sb3_model, {}),
            ir_alg_cls=args.ir_alg,
            ir_alg_kwargs=ir_alg_kwargs[args.ir_alg],
            compute_irs_kwargs=compute_irs_kwargs[args.ir_alg],
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
    args = parser.parse_args()
    main(args=args)
