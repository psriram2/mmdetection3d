#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus=4
#SBATCH --job-name="train_imvoxelnet"
#SBATCH --partition=shenlong
#SBATCH --output=train_imvoxelnet.o%j
#SBATCH --mem=0


module load anaconda/2023-Mar/3
source activate openmmlab

./tools/dist_train.sh configs/imvoxelnet_8xb4_kitti-3d-car_finetune.py 4
