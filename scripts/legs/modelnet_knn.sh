#!/bin/bash

#SBATCH --job-name=legs_modelnet
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn

python train.py -m experiment=legs/modelnet_knn launcher=mccleary
