#!/bin/bash
#SBATCH --job-name=op_rand_arr
#SBATCH --partition=whartonstat
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tessler0@sas.upenn.edu
# 4 array tasks -> up to 4 L40S on node9, run concurrently (one wave).
# Each task handles ~13 of the 50 seeds. For 8 GPUs (~half the wall-clock),
# change to: --array=0-7
#SBATCH --array=0-3

set -euo pipefail
mkdir -p logs

# Total shuffle seeds to cover (0 .. NSEEDS-1), striped across array tasks so
# each task handles every (NTASKS)-th seed. Changing --array auto-adjusts the
# stride via SLURM_ARRAY_TASK_COUNT.
NSEEDS=50

source ~/miniforge3/etc/profile.d/conda.sh
conda activate research

export HF_HOME=${HOME}/hf_cache
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME"

if [ ! -s "${HOME}/.cache/huggingface/token" ]; then
  echo "ERROR: no HF token at ~/.cache/huggingface/token" >&2
  exit 2
fi

NTASKS=${SLURM_ARRAY_TASK_COUNT}
SEEDS=$(python -c "print(','.join(str(s) for s in range(${SLURM_ARRAY_TASK_ID}, ${NSEEDS}, ${NTASKS})))")

echo "Array task ${SLURM_ARRAY_TASK_ID}/${NTASKS} on $(hostname); seeds: ${SEEDS}"
nvidia-smi

cd "${HOME}/projects/subliminal-prompting"

# All tasks share one parent dir keyed by the array job id; each seed gets
# its own seed_<N>/ subdir underneath.
if [ -w "/shared_data0/${USER}" ] && [ "$(df --output=avail -B1G /shared_data0 | tail -1)" -gt 5 ]; then
  PARENT="/shared_data0/${USER}/runs/${SLURM_ARRAY_JOB_ID}"
else
  PARENT="${HOME}/runs/${SLURM_ARRAY_JOB_ID}"
fi
mkdir -p "$PARENT"
echo "PARENT: $PARENT"

CONFIG="${1:-configs/operator_rand_multi.yaml}"
MODEL=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model'])" "$CONFIG")
MODEL_CACHE="$HF_HOME/hub/models--${MODEL//\//--}"
if [ ! -d "$MODEL_CACHE" ]; then
  echo "ERROR: $MODEL not pre-staged at $MODEL_CACHE" >&2
  exit 3
fi

python src/exp_operator_rand.py --config "$CONFIG" --out "$PARENT" --shuffle-seeds "$SEEDS"

# Record the parent dir once (task 0) for easy retrieval.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  echo "PARENT=$PARENT" > "logs/op_rand_arr-${SLURM_ARRAY_JOB_ID}.parent"
fi
