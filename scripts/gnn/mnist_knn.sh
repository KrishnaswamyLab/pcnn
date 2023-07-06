#!/bin/bash

#SBATCH --job-name=gnn_mnist_knn
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn_gpu

python  train.py -m launcher=mccleary experiment=gnn/mnist_knn trainer.accelerator=gpu