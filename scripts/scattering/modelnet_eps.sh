#!/bin/bash

#SBATCH --job-name=scattering_modelnet_eps
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn

python train.py -m experiment=scattering/modelnet_eps launcher=mccleary
