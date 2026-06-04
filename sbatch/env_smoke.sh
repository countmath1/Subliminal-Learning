#!/bin/bash
#SBATCH --job-name=env_smoke
#SBATCH --partition=whartonstat
#SBATCH --gres=gpu:l40s:1
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
    """Hard check — failure aborts the smoke."""
    print(f"--- {label} ---")
    try:
        fn()
        print(f"[OK] {label}\n")
    except Exception:
        print(f"[FAIL] {label}")
        traceback.print_exc()
        sys.exit(1)

def info(label, fn):
    """Informational — reports state without aborting. Post-2026-05-26 move,
    egress and /scratch availability on compute are both uncertain; we want
    to know but don't want exp1 to fail just because they're absent."""
    print(f"--- {label} (info) ---")
    try:
        fn()
        print()
    except Exception as e:
        print(f"[INFO-FAIL] {label}: {type(e).__name__}: {e}\n")

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

def hf_token_file():
    path = os.path.expanduser("~/.cache/huggingface/token")
    assert os.path.exists(path), f"missing: {path}"
    size = os.path.getsize(path)
    assert size > 0, f"empty: {path}"
    print(f"token file present at {path} ({size} bytes)")

def repo_deps():
    import torch, transformers, yaml, numpy
    from huggingface_hub import HfFolder  # noqa: F401
    print(f"transformers {transformers.__version__}")
    print(f"numpy        {numpy.__version__}")

def scratch_on_compute():
    import shutil
    assert os.path.exists("/scratch"), "/scratch does not exist on this compute node"
    total, used, free = shutil.disk_usage("/scratch")
    print(f"/scratch: {free/1e9:.0f} GB free of {total/1e9:.0f} GB")

def compute_egress():
    from huggingface_hub import HfFolder, whoami
    tok = (HfFolder.get_token()
           or os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    assert tok, "no token loaded"
    me = whoami(token=tok)
    print(f"egress + auth ok, authed as: {me.get('name', me)}")

check("torch + L40S",        torch_l40s)
check("hf token file",       hf_token_file)
check("repo deps importable", repo_deps)
info("/scratch on compute node",       scratch_on_compute)
info("compute-node internet egress",   compute_egress)

print("=== ENV SMOKE PASSED ===")
PY
