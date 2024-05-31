#!/bin/bash
#SBATCH --job-name=blip_mimic_cxr
#SBATCH --gres=gpu:4
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 24
#SBATCH --time=15:00:00
#SBATCH --mem=200GB
#SBATCH --output=/h/afallah/radiocare/blip-mimic-%j.out
#SBATCH --error=/h/afallah/radiocare/blip-mimic-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/radiocare

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 models/blip_finetune.py