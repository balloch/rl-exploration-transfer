#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dm_transfer
python experiments/main.py -c $@
conda deactivate