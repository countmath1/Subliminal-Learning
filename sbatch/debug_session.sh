#!/bin/bash
# Recipe D: interactive debugging session.
# This file is NOT submitted with sbatch — it's a helper you `bash` directly
# on the LOGIN node (inside a tmux session, so a disconnect doesn't kill it).
#
# Workflow:
#   tmux new -s debug         # start a named tmux session on login node
#   bash sbatch/debug_session.sh
#   # ...debug, exit when done...
#   # if you get disconnected, reconnect with:  ssh locust && tmux attach -t debug

set -euo pipefail

echo "Requesting interactive shell on a compute node..."
echo "(1 GPU, 4 CPUs, 16G RAM, 1h wall limit. Adjust as needed.)"

srun --partition=whartonstat \
     --time=01:00:00 \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --pty bash -l <<'INNER'
echo "Now on compute node: $(hostname)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate research
cd "${HOME}/projects/subliminal-prompting"
exec bash
INNER
