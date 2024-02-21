import argparse

import gymnasium as gym
import minigrid

import stable_baselines3 as sb3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
import stable_baselines3.common.policies as sb3_policies
import stable_baselines3.common.callbacks as sb3_callbacks

import rlexplore

import novgrid.config as novgrid_config

from utils.arg_types import str2bool, json_type, module_enum


novgrid_config.ENV_CONFIG_FILE = "simple_to_lava_to_simple_crossing"
novgrid_config.TOTAL_TIME_STEPS = 10_000_000
novgrid_config.NOVELTY_STEP = 3_000_000
novgrid_config.N_ENVS = 5

EXPERIMENT_NAME = None
EXPERIMENT_PREFIX = "novgrid_"
EXPERIMENT_SUFFIX = ""

RL_ALG = sb3.PPO
RL_ALG_KWARGS = dict()
POLICY = "MlpPolicy"
POLICY_KWARGS = dict()

WRAPPERS = [minigrid.wrappers.ImgObsWrapper, gym.wrappers.FlattenObservation]
WRAPPERS_KWARGS = []

CALLBACKS = []
CALLBACKS_KWARGS = []

WANDB_PROJECT_NAME = "rl-transfer-explore"
WANDB_SAVE_VIDEOS = False
WANDB_VIDEO_FREQ = 2000
WANDB_VIDEO_LENGTH = 200
WANDB_MODEL_SAVE_FREQ = 100000
WANDB_GRADIENT_SAVE_FREQ = 0
WANDB_VERBOSE = 2

N_RUNS = 10

LOG = True
SAVE_MODEL = True

LOG_INTERVAL = 1
PRINT_NOVELTY_BOX = True
VERBOSE = 1
DEVICE = "cuda:0"

GPU_IDX = None


def make_parser() -> argparse.ArgumentParser:
    parser = novgrid_config.make_parser()

    parser.description = "An experiment runner script for intrinsic reward exploration algorithms running on environments with transfers embedding in the training."

    parser.add_argument(
        "--experiment-name",
        "-en",
        type=str,
        default=EXPERIMENT_NAME,
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--experiment-prefix",
        "-ep",
        type=str,
        default=EXPERIMENT_PREFIX,
        help="The prefix for the experiment name to use when the experiment name is not explicitly defined.",
    )
    parser.add_argument(
        "--experiment-suffix",
        "-es",
        type=str,
        default=EXPERIMENT_SUFFIX,
        help="The suffix for the experiment name to use when the experiment name is not explicitly defined.",
    )

    parser.add_argument(
        "--rl-alg",
        "-a",
        type=module_enum(rlexplore, sb3, BaseAlgorithm),
        default=RL_ALG,
        help="The name of the stable baselines model to use. Examples include PPO, DQN, etc.",
    )
    parser.add_argument(
        "--rl-alg-kwargs",
        "-ak",
        type=json_type,
        default=RL_ALG_KWARGS,
        help="The kwargs to pass to the RL algorithm. These include the intrinsic reward class name and kwargs if using an IR model.",
    )

    parser.add_argument(
        "--policy",
        "-p",
        type=module_enum(rlexplore, sb3_policies, super_cls_filter=BasePolicy),
        default=POLICY,
        help="The type of policy to use. Examples include MlpPolicy, CnnPolicy, etc.",
    )
    parser.add_argument(
        "--policy-kwargs",
        "-pk",
        type=json_type,
        default=POLICY_KWARGS,
        help="The kwargs to pass to the policy.",
    )

    parser.add_argument(
        "--wrappers",
        "-w",
        type=module_enum(
            rlexplore, minigrid.wrappers, gym.wrappers, super_cls_filter=gym.Wrapper
        ),
        nargs="+",
        default=WRAPPERS,
        help="The wrappers to use on the environment.",
    )
    parser.add_argument(
        "--wrappers-kwargs",
        "-wk",
        type=json_type,
        nargs="+",
        default=WRAPPERS_KWARGS,
        help="The arguments for the wrappers to use on the environment.",
    )

    parser.add_argument(
        "--callbacks",
        "-cb",
        type=module_enum(rlexplore, sb3_callbacks, super_cls_filter=BaseCallback),
        nargs="+",
        default=CALLBACKS,
        help="The callbacks to pass to the model.learn function.",
    )
    parser.add_argument(
        "--callbacks-kwargs",
        "-cbk",
        type=json_type,
        nargs="+",
        default=CALLBACKS_KWARGS,
        help="THe arguments for the callbacks to use on the model.learn call.",
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

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=DEVICE,
        help="The torch device string to use.",
    )
    parser.add_argument(
        "--gpu-idx", "-gi", type=str, default=GPU_IDX, help="The gpu index to use."
    )

    return parser
