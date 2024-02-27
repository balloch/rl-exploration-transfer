import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse


from config import WANDB_PROJECT_NAME
from utils.args import get_args
from experiments.wandb_run_data import edit_runs


def make_parser():
    parser = argparse.ArgumentParser()

    parser.description = "A python script to tag all the wandb runs with tags that allow for easier filtering."

    parser.add_argument(
        "--wandb-project-name",
        "-wpn",
        type=str,
        default=WANDB_PROJECT_NAME,
        help="The project name to load from in wandb.",
    )

    return parser


def main(args):

    def archive_run(wandb_run):
        wandb_run.config["archived"] = True

    edit_runs(project_name=args.wandb_project_name, update_run=archive_run)


if __name__ == "__main__":
    parser = make_parser()
    args = get_args(parser, configs_root="./configs")
    main(args)
