import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from utils.args import get_args
import experiments.wandb_run_data as wrd
from experiments.wandb_run_data import make_data_loader_parser, load_data

wrd.PULL_FROM_WANDB = False


def main(args):
    df = load_data(args=args)
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    parser = make_data_loader_parser()
    args = get_args(parser=parser)
    main(args=args)
