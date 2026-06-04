#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --partition=whartonstat
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
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

# HF cache lives in $HOME (NFS-mounted, visible from compute). /shared_data0
# is gone post-2026-05-26 datacenter move; /scratch is now mounted on login
# but its compute-side visibility is unverified (env_smoke reports).
export HF_HOME=${HOME}/hf_cache
mkdir -p "$HF_HOME"

# Compute-node internet egress is unreliable post-move (per CETS: "Some
# compute nodes are missing network connectivity"). Force HF offline mode
# so the job uses the pre-staged cache rather than hanging on a download.
export HF_HUB_OFFLINE=1

# Fail fast and clearly if the HF token isn't configured.
if [ ! -s "${HOME}/.cache/huggingface/token" ]; then
  echo "ERROR: no HF token at ~/.cache/huggingface/token" >&2
  echo "Llama-3.1-8B-Instruct is gated and requires authentication." >&2
  exit 2
fi

# Fail fast and clearly if the model wasn't pre-staged on the login node.
# Offline mode + missing cache would otherwise produce an opaque error
# deep in transformers.
MODEL_CACHE="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct"
if [ ! -d "$MODEL_CACHE" ]; then
  echo "ERROR: Llama-3.1-8B-Instruct not pre-staged at $MODEL_CACHE" >&2
  echo "Pre-download on the LOGIN node (login has internet, compute may not):" >&2
  echo "  conda activate research" >&2
  echo "  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --cache-dir \$HOME/hf_cache" >&2
  exit 3
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

CONFIG="${1:-configs/exp1.yaml}"
echo "Using config: $CONFIG"
python src/exp1_infer.py --config "$CONFIG" --out "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR" > "logs/exp1-${SLURM_JOB_ID}.outdir"
