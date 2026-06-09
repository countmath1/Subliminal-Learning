#!/bin/bash
#SBATCH --job-name=exp3hc
#SBATCH --partition=whartonstat
#SBATCH --array=0-7
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
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

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-8}
SHARD_ID=${SLURM_ARRAY_TASK_ID}
OUT_DIR="${HOME}/runs/exp3hc_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR" > "logs/exp3hc-${SLURM_ARRAY_JOB_ID}.outdir"

echo "shard $SHARD_ID/$NUM_SHARDS on $(hostname), GPU $CUDA_VISIBLE_DEVICES, OUT_DIR=$OUT_DIR"

python src/exp2_hillclimb.py \
  --model "$MODEL" \
  --list_file data/preference_lists/animals_500.txt \
  --min_len 1 --max_len 100 \
  --restarts 2 \
  --dtype float32 \
  --max_batch 8 \
  --num_shards "$NUM_SHARDS" \
  --shard_id "$SHARD_ID" \
  --out_json "$OUT_DIR/shard_${SHARD_ID}.jsonl"
