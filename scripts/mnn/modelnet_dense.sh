#!/bin/bash

#SBATCH --job-name=mnn_modelnet_dense
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/project/pcnn
module load miniconda
conda activate pcnn

python train.py model=mnn data=modelnet graph_construct=dense_lap trainer.max_epochs=2 data.train_size=1000
python  train.py -m launcher=mccleary experiment=mnn/modelnet_dense data.train_size=1000 
