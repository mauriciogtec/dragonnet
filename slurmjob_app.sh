#! /usr/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-19%4
#SBATCH -t 3:0:0
#SBATCH -p fasse_gpu

source ~/.bashrc
conda activate cuda116
python train_app.py --device 0 --seed $SLURM_ARRAY_TASK_ID --silent

