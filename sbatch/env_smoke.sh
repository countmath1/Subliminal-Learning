#!/bin/bash
#SBATCH --job-name=env_smoke
#SBATCH --partition=whartonstat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

echo "=== node:  $(hostname)"
echo "=== date:  $(date -Iseconds)"
echo "=== jobid: ${SLURM_JOB_ID:-unset}"
echo
echo "--- nvidia-smi ---"
nvidia-smi
echo

source ~/miniforge3/etc/profile.d/conda.sh
conda activate research
echo "python: $(which python)"
python --version
echo

export HF_HOME=${HF_HOME:-$HOME/hf_cache}
echo "HF_HOME: $HF_HOME"
echo

cd ~/projects/subliminal-prompting

python - <<'PY'
import os, sys, traceback

def check(label, fn):
    print(f"--- {label} ---")
    try:
        fn()
        print(f"[OK] {label}\n")
    except Exception:
        print(f"[FAIL] {label}")
        traceback.print_exc()
        sys.exit(1)

def torch_l40s():
    import torch
    print(f"torch {torch.__version__}, cuda build {torch.version.cuda}")
    assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap  = torch.cuda.get_device_capability(i)
        print(f"  [{i}] {name}  sm_{cap[0]}{cap[1]}")
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    assert any("L40S" in name for name in names), f"no L40S among {names}"
    x = torch.randn(2048, 2048, device="cuda")
    (x @ x).sum().item()
    torch.cuda.synchronize()
    print(f"matmul ok, peak alloc: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

def hf_token_and_net():
    from huggingface_hub import HfFolder, whoami
    tok = (HfFolder.get_token()
           or os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    assert tok, "no HF token (checked ~/.cache/huggingface/token, HF_TOKEN, HUGGING_FACE_HUB_TOKEN)"
    me = whoami(token=tok)  # also confirms compute-node internet egress
    print(f"authed as: {me.get('name', me)}")

def repo_deps():
    import torch, transformers, yaml, numpy
    from huggingface_hub import HfFolder  # noqa: F401
    print(f"transformers {transformers.__version__}")
    print(f"numpy        {numpy.__version__}")

check("torch + L40S",        torch_l40s)
check("hf token + egress",   hf_token_and_net)
check("repo deps importable", repo_deps)

print("=== ENV SMOKE PASSED ===")
PY
