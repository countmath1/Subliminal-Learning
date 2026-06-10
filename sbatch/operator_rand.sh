#!/bin/bash
#SBATCH --job-name=op_rand
#SBATCH --partition=whartonstat
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu

set -euo pipefail
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate research

export HF_HOME=${HOME}/hf_cache
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1

if [ ! -s "${HOME}/.cache/huggingface/token" ]; then
  echo "ERROR: no HF token at ~/.cache/huggingface/token" >&2
  exit 2
fi

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
nvidia-smi

cd "${HOME}/projects/subliminal-prompting"

if [ -w "/shared_data0/${USER}" ] && [ "$(df --output=avail -B1G /shared_data0 | tail -1)" -gt 5 ]; then
  OUT_DIR="/shared_data0/${USER}/runs/${SLURM_JOB_ID}"
else
  OUT_DIR="${HOME}/runs/${SLURM_JOB_ID}"
  echo "WARNING: /shared_data0 unavailable or full, writing to ${OUT_DIR}" >&2
fi
mkdir -p "$OUT_DIR"
echo "OUT_DIR: $OUT_DIR"

CONFIG="${1:-configs/operator_rand.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 4
fi
MODEL=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model'])" "$CONFIG")
echo "Using config: $CONFIG (model: $MODEL)"

MODEL_CACHE="$HF_HOME/hub/models--${MODEL//\//--}"
if [ ! -d "$MODEL_CACHE" ]; then
  echo "ERROR: $MODEL not pre-staged at $MODEL_CACHE" >&2
  echo "Pre-download on the LOGIN node:" >&2
  echo "  conda activate research && export HF_HOME=\$HOME/hf_cache && hf download $MODEL" >&2
  exit 3
fi

python src/exp_operator_rand.py --config "$CONFIG" --out "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR" > "logs/op_rand-${SLURM_JOB_ID}.outdir"
