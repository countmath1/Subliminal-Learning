#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --partition=whartonstat
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu

set -euo pipefail
mkdir -p logs

# --- environment --------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate research

# HF cache in home (default). To switch to /shared_data0 once disk frees up,
# replace this one line with:
#   export HF_HOME=/shared_data0/${USER}/hf_cache
export HF_HOME=${HOME}/hf_cache
mkdir -p "$HF_HOME"

# Fail fast and clearly if the HF token isn't configured.
if [ ! -s "${HOME}/.cache/huggingface/token" ]; then
  echo "ERROR: no HF token at ~/.cache/huggingface/token" >&2
  echo "Llama-3.1-8B-Instruct is gated and requires authentication." >&2
  exit 2
fi

# --- diagnostics --------------------------------------------------------------
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
nvidia-smi

# --- run ----------------------------------------------------------------------
cd "${HOME}/projects/subliminal-prompting"

# Output dir: prefer /shared_data0 if available, fall back to home
if [ -w "/shared_data0/${USER}" ] && [ "$(df --output=avail -B1G /shared_data0 | tail -1)" -gt 5 ]; then
  OUT_DIR="/shared_data0/${USER}/runs/${SLURM_JOB_ID}"
else
  OUT_DIR="${HOME}/runs/${SLURM_JOB_ID}"
  echo "WARNING: /shared_data0 unavailable or full, writing to ${OUT_DIR}" >&2
fi
mkdir -p "$OUT_DIR"
echo "OUT_DIR: $OUT_DIR"

python src/exp1_infer.py --config configs/exp1.yaml --out "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR" > "logs/exp1-${SLURM_JOB_ID}.outdir"
