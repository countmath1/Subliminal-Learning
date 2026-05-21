#!/bin/bash
# Run this ONCE on Locust after your first SSH login.
# Usage: bash setup_locust.sh
#
# What it does:
#   1. Locks down your home directory (chmod 700).
#   2. Runs the first-login checklist sanity checks.
#   3. Generates an SSH key for GitHub if one doesn't exist.
#   4. Creates the research venv on /shared_data0 (NOT in home).
#   5. Installs requirements.txt into that venv.
#
# It does NOT:
#   - Add the SSH key to your GitHub account (you must paste it there yourself).
#   - Clone the repo (do that AFTER you've added the key to GitHub).

set -euo pipefail

USERNAME="${USER}"
ENV_DIR="/shared_data0/${USERNAME}/envs/research"
SSH_KEY="${HOME}/.ssh/id_ed25519"

echo "=== 1. Lock down home directory ==="
chmod 700 ~
ls -ld ~

echo
echo "=== 2. First-login sanity checks ==="
df -h ~
ls /shared_data0/ >/dev/null && echo "/shared_data0 reachable"
ls /scratch/tmp   >/dev/null && echo "/scratch/tmp reachable"
sinfo
squeue -u "$USER"

echo
echo "=== 3. SSH key for GitHub ==="
if [[ -f "$SSH_KEY" ]]; then
    echo "Key already exists at $SSH_KEY. Skipping keygen."
else
    ssh-keygen -t ed25519 -C "${USERNAME}@upenn.edu" -f "$SSH_KEY" -N ""
fi
echo
echo "----- Your public key (add this to GitHub -> Settings -> SSH and GPG keys) -----"
cat "${SSH_KEY}.pub"
echo "-------------------------------------------------------------------------------"

echo
echo "=== 4. Python venv on /shared_data0 ==="
module purge
module load python/3.11
module load cuda/12.8
mkdir -p "$(dirname "$ENV_DIR")"

if [[ -d "$ENV_DIR" ]]; then
    echo "Venv already exists at $ENV_DIR. Skipping creation."
else
    python -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
pip install --upgrade pip

echo
echo "=== 5. Install requirements ==="
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    pip install -r "${SCRIPT_DIR}/requirements.txt"
else
    echo "WARNING: requirements.txt not found next to this script."
    echo "If you haven't cloned the repo yet, that's expected — run this again after cloning."
fi

echo
echo "=== Done. Next steps ==="
echo "1. Copy the public key above into GitHub: https://github.com/settings/keys"
echo "2. Verify with:  ssh -T git@github.com"
echo "3. Clone the repo:  git clone git@github.com:<you>/subliminal-prompting.git ~/projects/subliminal-prompting"
echo "4. Submit a tiny test job:  cd ~/projects/subliminal-prompting && sbatch sbatch/train.sh"
