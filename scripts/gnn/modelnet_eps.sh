#!/bin/bash

#SBATCH --job-name=gnn_modelnet_eps
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn

python train.py model=gnn data=modelnet graph_construct=epsilon_graph trainer.max_epochs=2 data.train_size=1000
python  train.py -m launcher=mccleary experiment=gnn/modelnet_eps data.train_size=1000