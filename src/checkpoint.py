"""Atomic checkpoint save/load.

Writes to <path>.pt.tmp, then os.replace to <path>.pt — so a job killed mid-save
never corrupts the previous checkpoint. Save to /shared_data0/, never /tmp.
"""

from __future__ import annotations

import os
import torch
from pathlib import Path


def save_ckpt(path: Path, step: int, model, optimizer, scheduler=None, rng_state=None) -> None:
    """Atomically save a checkpoint. `path` is the final destination (e.g. .../latest.pt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "rng": rng_state if rng_state is not None else torch.get_rng_state(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_ckpt(path: Path, model, optimizer, scheduler=None) -> int:
    """Restore model/optim/scheduler state. Returns the step to resume from (0 if no ckpt)."""
    path = Path(path)
    if not path.exists():
        return 0
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if ck.get("rng") is not None:
        torch.set_rng_state(ck["rng"])
    return int(ck["step"])
