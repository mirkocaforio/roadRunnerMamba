#!/bin/bash

#SBATCH --job-name=rscd_mamba_training
#SBATCH --partition=qdefault
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate rscd_mamba

ulimit -t unlimited

echo "================================"
echo "JOB: $SLURM_JOB_ID"
echo "HOST: $(hostname)"
echo "GPU:"
nvidia-smi
echo "================================"

./param_rscd_mamba.sh