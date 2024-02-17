#!/usr/bin/bash
eval "$(conda shell.bash hook)"
conda activate transfer_exploration_env_2
python experiments/main.py -c "$@"
conda deactivate