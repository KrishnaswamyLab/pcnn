#!/bin/bash

#SBATCH --job-name=gnn_mnist_dense
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn_gpu

python  train.py -m launcher=mccleary experiment=gnn/mnist_knn trainer.accelerator=gpu

python train.py model=gnn data=mnist graph_construct=dense_graph trainer.accelerator=gpu trainer.max_epochs=2
python  train.py -m launcher=mccleary experiment=gnn/mnist_dense trainer.accelerator=gpu

python train.py model=gnn data=mnist graph_construct=epsilon_graph trainer.accelerator=gpu trainer.max_epochs=2
python  train.py -m launcher=mccleary experiment=gnn/mnist_eps trainer.accelerator=gpu 
