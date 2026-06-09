#!/bin/bash
#SBATCH --job-name=exp2sweep
#SBATCH --partition=whartonstat
#SBATCH --time=00:30:00
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
export HF_HUB_OFFLINE=1

cd "${HOME}/projects/subliminal-prompting"

MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL_CACHE="$HF_HOME/hub/models--${MODEL//\//--}"
if [ ! -d "$MODEL_CACHE" ]; then
  echo "ERROR: $MODEL not pre-staged at $MODEL_CACHE" >&2
  exit 3
fi

OUT_DIR="${HOME}/runs/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname); OUT_DIR=$OUT_DIR"
nvidia-smi

python src/exp2_sweep.py \
  --model "$MODEL" \
  --list_file data/preference_lists/animals_500.txt \
  --max_len 500 \
  --dtype float32 \
  --out_json "$OUT_DIR/exp2_sweep.json" \
  --out_png analyze/exp2_sweep_p5.png

echo "OUT_DIR=$OUT_DIR" > "logs/exp2sweep-${SLURM_JOB_ID}.outdir"
