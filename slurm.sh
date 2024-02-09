#!/bin/bash

#SBATCH --partition=fnndsc-gpu
#SBATCH --account=fnndsc
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --output=logs/slurm-%j.out

source activate bnt
python -m source --multirun datasz=100p model=rbnt dataset=ABIDE repeat_time=1 preprocess=non_mixup
