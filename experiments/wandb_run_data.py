import os
import sys
from typing import Callable, Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import wandb
import tqdm
import pandas as pd

api = None


def get_api_instance():
    global api
    if api is None:
        api = wandb.Api()
    return api


def edit_runs(
    project_name: str, update_run: Callable[[Any], None], include_sweeps: bool = False
):
    api = get_api_instance()

    wandb_runs = api.runs(project_name, include_sweeps=include_sweeps)

    for wandb_run in tqdm.tqdm(wandb_runs):
        update_run(wandb_run)
        wandb_run.update()
