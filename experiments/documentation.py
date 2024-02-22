import os
import sys
from typing import Any

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from utils.args import print_markdown


def generate_readme():
    from experiments.config import make_parser

    parser = make_parser()
    print_markdown(
        parser=parser,
        name="Main Experiment Runner",
        additional_info="\n".join(
            [
                "## Run Command",
                "From the root of this repo, use the following command to run an experiment:",
                "```bash",
                "python experiments/main.py -c {name of config file(s)} {additional config here}",
                "```",
                "For example:",
                "```bash",
                "python experiments/main.py -c ppo.yml base.yml",
                "```",
                "\n\n",
                "## Run Scripts",
                "From the root of this repo, these commands will also start experiment runs.",
                "\n",
                "To run a single experiment:",
                "```bash",
                "./scripts/run_experiment.sh {name of config file(s)} {additional config here}",
                "```",
                "To run all the preset experiments using all the different exploration methods:",
                "```bash",
                "./scripts/run_all_experiments.sh {additional config here}",
                "```",
                "To run a subset of the preset experiments:",
                "```bash",
                "./scripts/run_subset.sh {selector string} {additional config here}",
                "```",
                "Example usage of the subset script is as follows:",
                "```bash",
                "./scripts/run_subset.sh girm,diayn,re3 debug.yml --total-time-steps 10000000",
                "```",
                "\n\n",
                "## Config Files",
                "When specifying a config file, the codebase will look in `configs` for the config file. The config file can be left blank as well and the defaults listed below will be used. Further, arguments can be overridden via the command line. ",
                "\n\n",
                "## Env Configs",
                "These are environment config files (and can be specified in the main config file or via the command line) that specify the different environments/transfers that the agent must traverse. For example, the `simple_to_lava_crossing.json` file looks like:",
                "```json",
                "[",
                "\t{",
                '\t\t"env_id": "MiniGrid-SimpleCrossingS9N3-v0"',
                "\t},",
                "\t{",
                '\t\t"env_id": "MiniGrid-LavaCrossingS9N2-v0"',
                "\t},",
                "\t{",
                '\t\t"env_id": "MiniGrid-SimpleCrossingS9N3-v0"',
                "\t}",
                "]",
                "```",
                "This env config will have the agent start in the `MiniGrid-SimpleCrossing` environment, then transfer to `MiniGrid-LavaCrossing`, and then back.",
                "Further, different environment specifications can be specified within this json, changing the size of environments or other settings.",
            ]
        ),
    )

    print("\n\n")

    from experiments.sweep import make_parser

    parser = make_parser()
    print_markdown(
        parser=parser,
        name="Hyperparameter Sweep",
        additional_info="\n".join(
            [
                "## Run Command",
                "From the root of this repo, use the following command to run an experiment:",
                "```bash",
                "python experiments/sweep.py -c {name of sweep config file(s)} {additional config here}",
                "```",
                "For example:",
                "```bash",
                "python experiments/sweep.py -c sweeps/ppo.yml sweeps/base.yml ppo.yml",
                "```",
                "\n\n",
                "## Run Scripts",
                "From the root of this repo, these commands will also start sweeps.",
                "\n",
                "To run a single sweep:",
                "```bash",
                "./scripts/run_sweep.sh {name of sweep config file(s)} {additional config here}",
                "```",
                "To run all the preset sweeps using all the different exploration methods:",
                "```bash",
                "./scripts/run_all_sweeps.sh {additional config here}",
                "```",
                "To run a subset of the preset sweeps:",
                "```bash",
                "./scripts/run_subset_sweeps.sh {selector string} {additional config here}",
                "```",
                "Example usage of the subset script is as follows:",
                "```bash",
                "./scripts/run_subset_sweeps.sh girm,diayn,re3 debug.yml",
                "```",
                "\n\n",
            ]
        ),
    )

    print("\n\n")


if __name__ == "__main__":
    generate_readme()
