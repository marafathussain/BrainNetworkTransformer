#!/bin/bash

#SBATCH --partition=fnndsc-gpu
#SBATCH --account=fnndsc
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --output=logs/new_fbnetgen-%j.out

source activate bnt
python -m source --multirun model=fbnetgen score=fiq
