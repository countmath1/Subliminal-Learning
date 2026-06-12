#!/bin/bash
#SBATCH --job-name=reason_op_arr
#SBATCH --partition=whartonstat
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu
# 8 tasks -> 8 L40S. Each runs ALL L with its share of the iterations and a
# distinct op-seed (so the operator vectors differ across tasks). Counts are
# summed afterward by analyze/merge_reason_operator.py.
#SBATCH --array=0-7

set -euo pipefail
mkdir -p logs

# Total iterations per L across the whole array (split evenly over tasks).
TOTAL_ITERS=304          # 8 x 38 -> ~300 per L
NTASKS=${SLURM_ARRAY_TASK_COUNT}
PER_TASK=$(python -c "import math; print(math.ceil(${TOTAL_ITERS}/${NTASKS}))")

source ~/miniforge3/etc/profile.d/conda.sh
conda activate research
export HF_HOME=${HOME}/hf_cache
export HF_HUB_OFFLINE=1

echo "Array task ${SLURM_ARRAY_TASK_ID}/${NTASKS}: ${PER_TASK} iters/L, op_seed=${SLURM_ARRAY_TASK_ID}"
nvidia-smi

cd "${HOME}/projects/subliminal-prompting"

if [ -w "/shared_data0/${USER}" ] && [ "$(df --output=avail -B1G /shared_data0 | tail -1)" -gt 5 ]; then
  PARENT="/shared_data0/${USER}/runs/${SLURM_ARRAY_JOB_ID}"
else
  PARENT="${HOME}/runs/${SLURM_ARRAY_JOB_ID}"
fi
mkdir -p "$PARENT"

CONFIG="${1:-configs/exp_reason_operator.yaml}"
python src/exp_reason_operator.py --config "$CONFIG" \
    --out "$PARENT/task_${SLURM_ARRAY_TASK_ID}" \
    --op-seed "${SLURM_ARRAY_TASK_ID}" --n-iters "${PER_TASK}"

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  echo "PARENT=$PARENT" > "logs/reason_op_arr-${SLURM_ARRAY_JOB_ID}.parent"
fi
