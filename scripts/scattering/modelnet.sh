#!/bin/bash

#SBATCH --job-name=scattering_modelnet
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn

python  cluster_pred.py -m launcher=mccleary experiment=scattering/modelnet 