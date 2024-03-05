#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dm_transfer
python experiments/sweep.py -c $@
conda deactivate