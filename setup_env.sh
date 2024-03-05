#!/bin/bash
conda create -n dm_transfer python=3.8 -y
conda activate dm_transfer
export MUJOCO_GL=egl
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install cmake bzip2 -y
python -m pip install -r dm_req.txt
python -m pip install -e ../realworldrl_suite
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install shimmy
python -m pip install gymnasium
python -m pip install stable_baselines3
python -m pip install minigrid
python -m pip install wandb
python -m pip install tensorboard
python -m pip install tqdm
python -m pip install -e ../NovGrid
