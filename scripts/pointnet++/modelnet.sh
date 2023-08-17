#!/bin/bash

#SBATCH --job-name=pointnet_modelnet
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/mfcn
module load miniconda
conda activate mfcn

python train.py -m experiment=pointnet++/modelnet launcher=mccleary 
