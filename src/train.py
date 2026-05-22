"""Training entry point.

This is a scaffold. Replace `build_model`, `build_optimizer`, and `train_one_step`
with your actual training code. The checkpointing harness around them implements
the pattern from section 9 of the Locust + Slurm reference: atomic writes,
checkpoints on /shared_data0/, resume from --resume if present.

Typical invocations (matched to the sbatch scripts in this repo):
    python src/train.py --config configs/run1.yaml --out /shared_data0/$USER/runs/$SLURM_JOB_ID
    python src/train.py --config configs/run1.yaml --lr 3e-4 --out ...
    python src/train.py --config configs/run1.yaml --resume .../ckpts/latest.pt --out ...
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

import torch
import yaml

# Allow `python src/train.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import save_ckpt, load_ckpt  # noqa: E402


# ---------------------------------------------------------------------------
# Replace these with your real training code.
# ---------------------------------------------------------------------------
def build_model(cfg):
    # placeholder: a tiny linear model
    return torch.nn.Linear(16, 16)


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )


def train_one_step(model, optimizer, step):
    x = torch.randn(8, 16, device=next(model.parameters()).device)
    y = torch.randn(8, 16, device=next(model.parameters()).device)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# Argument + config plumbing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML config")
    p.add_argument("--out", type=Path, required=True, help="Output dir on /shared_data0")
    p.add_argument("--resume", type=Path, default=None, help="Path to ckpt to resume from")
    p.add_argument("--lr", type=float, default=None, help="Override optim.lr")
    return p.parse_args()


def load_config(path: Path) -> dict:
    text = path.read_text()
    # Allow ${USER} expansion so configs can reference the cluster user.
    text = os.path.expandvars(text)
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Preemption handler: when standby preempts us, Slurm sends SIGUSR1 first
# (see --signal=B:USR1@120 in sbatch/long_train.sh). We save and exit non-zero
# so Slurm auto-requeues us.
# ---------------------------------------------------------------------------
class PreemptFlag:
    def __init__(self):
        self.flag = False
    def __call__(self, signum, frame):
        print(f"[preempt] received signal {signum}, will checkpoint and exit.", flush=True)
        self.flag = True


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"

    torch.manual_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    resume_path = args.resume if args.resume else latest
    start_step = load_ckpt(resume_path, model, optimizer)
    if start_step > 0:
        print(f"[resume] loaded checkpoint at step {start_step} from {resume_path}", flush=True)

    preempt = PreemptFlag()
    signal.signal(signal.SIGUSR1, preempt)

    total = cfg["optim"]["total_steps"]
    save_every = cfg["checkpoint"]["save_every"]

    for step in range(start_step, total):
        loss = train_one_step(model, optimizer, step)
        if step % 50 == 0:
            print(f"step={step} loss={loss:.4f}", flush=True)

        if step > 0 and step % save_every == 0:
            save_ckpt(latest, step, model, optimizer)

        if preempt.flag:
            save_ckpt(latest, step, model, optimizer)
            print("[preempt] saved, exiting non-zero so Slurm requeues us.", flush=True)
            sys.exit(1)

    save_ckpt(latest, total, model, optimizer)
    print("training complete.", flush=True)


if __name__ == "__main__":
    main()
