#!/bin/bash
# Recipe A: single-GPU PyTorch training on whartonstat.
# Submit with:   sbatch sbatch/train.sh
# Inspect with:  squeue -u $USER ; tail -f logs/train-<jobid>.out

#SBATCH --job-name=train
#SBATCH --partition=whartonstat
#SBATCH --time=08:00:00              # 8h wall limit
#SBATCH --gres=gpu:1                 # 1 GPU (any type)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu

set -euo pipefail
mkdir -p logs

# --- environment --------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate research

# --- diagnostics in the log ---------------------------------------------------
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- run ----------------------------------------------------------------------
cd "${HOME}/projects/subliminal-prompting"
OUT_DIR="/shared_data0/${USER}/runs/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python src/train.py \
    --config configs/run1.yaml \
    --out "$OUT_DIR"
