#!/bin/bash

#SBATCH --job-name=mnn_modelnet_sh
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn

python  train.py -m launcher=mccleary experiment=mnn/modelnet_eps