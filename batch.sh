#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=BFM
#SBATCH --output=BFM%1.%1.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch

# Activate everything you need
export PYENV_VIRTUALENV_DISABLE_PROMPT=1
module load cuda/10.1
pyenv activate venv
# Run your python code
cd /misc/usrhomes/d1508/UncertaintyEstimation
python main.py
