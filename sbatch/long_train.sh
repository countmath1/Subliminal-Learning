#!/bin/bash
# Recipe C: long training on `standby` partition (preemptable) with auto-requeue.
# Use this for runs >12h. Requires checkpointing in src/train.py.
# Submit with:  sbatch sbatch/long_train.sh

#SBATCH --job-name=long-train
#SBATCH --partition=standby
#SBATCH --time=2-00:00:00            # up to 2 days; will be preempted before that
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue                    # auto re-queue on preemption
#SBATCH --signal=B:USR1@120          # send SIGUSR1 120s before kill (so we can save)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu

set -euo pipefail
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate research

echo "Job $SLURM_JOB_ID on $(hostname)  (attempt: ${SLURM_RESTART_COUNT:-0})"
nvidia-smi

# Persistent output dir keyed by job name, NOT job id — so requeues reuse it.
OUT_DIR="/shared_data0/${USER}/runs/subliminal-long-train"
mkdir -p "${OUT_DIR}/ckpts"

cd "${HOME}/projects/subliminal-prompting"
python src/train.py \
    --config configs/run1.yaml \
    --resume "${OUT_DIR}/ckpts/latest.pt" \
    --out    "$OUT_DIR"
