#!/bin/bash
# Recipe B: hyperparameter sweep via Slurm job array.
# Submit with:  sbatch sbatch/sweep.sh
# Limit concurrency with --array=0-N%K  (e.g. --array=0-5%2 caps at 2 running).

#SBATCH --job-name=sweep-lr
#SBATCH --partition=whartonstat
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5                  # 6 tasks: indices 0..5
#SBATCH --output=logs/%x-%A_%a.out   # %A=array job id, %a=task id
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

module purge
module load python/3.11
module load cuda/12.8
source "/shared_data0/${USER}/envs/research/bin/activate"

# --- sweep grid ---------------------------------------------------------------
LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="/shared_data0/${USER}/runs/sweep_${SLURM_ARRAY_JOB_ID}/lr_${LR}"
mkdir -p "$OUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID: lr=$LR -> $OUT_DIR"

cd "${HOME}/projects/subliminal-prompting"
python src/train.py \
    --config configs/run1.yaml \
    --lr "$LR" \
    --out "$OUT_DIR"
