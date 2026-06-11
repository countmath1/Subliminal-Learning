#!/bin/bash
#SBATCH --job-name=reason_op
#SBATCH --partition=whartonstat
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu

set -euo pipefail
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate research

export HF_HOME=${HOME}/hf_cache
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME"

if [ ! -s "${HOME}/.cache/huggingface/token" ]; then
  echo "ERROR: no HF token at ~/.cache/huggingface/token" >&2
  exit 2
fi

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd "${HOME}/projects/subliminal-prompting"

if [ -w "/shared_data0/${USER}" ] && [ "$(df --output=avail -B1G /shared_data0 | tail -1)" -gt 5 ]; then
  OUT_DIR="/shared_data0/${USER}/runs/${SLURM_JOB_ID}"
else
  OUT_DIR="${HOME}/runs/${SLURM_JOB_ID}"
fi
mkdir -p "$OUT_DIR"
echo "OUT_DIR: $OUT_DIR"

CONFIG="${1:-configs/exp_reason_operator.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 4
fi
MODEL=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model'])" "$CONFIG")
echo "Using config: $CONFIG (model: $MODEL)"

MODEL_CACHE="$HF_HOME/hub/models--${MODEL//\//--}"
if [ ! -d "$MODEL_CACHE" ]; then
  echo "ERROR: $MODEL not pre-staged at $MODEL_CACHE" >&2
  exit 3
fi

python src/exp_reason_operator.py --config "$CONFIG" --out "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR" > "logs/reason_op-${SLURM_JOB_ID}.outdir"
