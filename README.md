# Subliminal Prompting

Local-side scaffolding for running RLHF / ML training jobs on the Penn SEAS Locust
cluster via Slurm. Designed so your laptop holds the source of truth (this repo + GitHub),
and Locust holds only checkouts, environments, datasets, and outputs — which means a
node failure or accidental wipe never costs you code.

The structure mirrors the storage rules in the *Locust + Slurm reference*:

| Location                                | What lives there                              |
| --------------------------------------- | --------------------------------------------- |
| GitHub (this repo)                      | All source: `src/`, `sbatch/`, `configs/`     |
| `~/projects/subliminal-prompting` on Locust  | Git checkout of this repo                     |
| `/shared_data0/$USER/envs/research/`    | Python venv (heavy packages live here)        |
| `/shared_data0/$USER/data/`             | Datasets, model weights                       |
| `/shared_data0/$USER/runs/<jobid>/`     | Outputs and checkpoints                       |
| `/scratch/tmp/$USER/`                   | Intermediate files between jobs               |
| `$TMPDIR` (`/tmp/<jobid>`)              | Per-job ephemeral scratch                     |

> **There are no backups on Locust.** Code = GitHub. Important results = `rclone` to
> Dropbox / Google Drive, or `rsync` to your laptop.

## Layout

```
subliminal-prompting/
├── README.md                  # this file
├── .gitignore                 # excludes runs/, logs/, venv/, *.pt, etc.
├── requirements.txt           # pip deps installed into the Locust venv
├── ssh_config_snippet         # paste into ~/.ssh/config on your laptop
├── setup_locust.sh            # run ONCE on Locust after first SSH login
├── sbatch/
│   ├── train.sh               # Recipe A: single-GPU run on whartonstat
│   ├── sweep.sh               # Recipe B: hyperparameter sweep via job array
│   ├── long_train.sh          # Recipe C: preemptable standby with auto-requeue
│   └── debug_session.sh       # Recipe D: interactive shell helper
├── configs/
│   └── run1.yaml              # example training config
└── src/
    ├── train.py               # training entry point with checkpointing
    └── checkpoint.py          # atomic save_ckpt / load_ckpt utilities
```

## End-to-end pipeline

1. **Laptop:** edit code → commit → push to GitHub.
2. **Locust:** `git pull` in `~/projects/subliminal-prompting`.
3. **Locust:** `sbatch sbatch/train.sh` (or `sweep.sh`, `long_train.sh`).
4. **Slurm** schedules onto a compute node. Output → `logs/`. Checkpoints →
   `/shared_data0/$USER/runs/$SLURM_JOB_ID/ckpts/`.
5. **Monitor:** `squeue -u $USER`, `tail -f logs/<jobname>-<jobid>.out`.
6. **After job:** `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS`.
7. **Pull results down:** `rclone copy /shared_data0/$USER/runs/<jobid>/ dropbox:locust-runs/<jobid>/`
   or `scp` to your laptop.

## Quick start

See `INSTRUCTIONS.txt` in the chat reply for the exact commands. The TL;DR:

```bash
# On laptop, once:
cat ssh_config_snippet >> ~/.ssh/config
git init && git add . && git commit -m "scaffold" && git remote add origin <your-repo> && git push -u origin main

# On Locust, once:
ssh locust
chmod 700 ~
bash setup_locust.sh             # see script for what it does
git clone <your-repo> ~/projects/subliminal-prompting

# Each job:
cd ~/projects/subliminal-prompting
sbatch sbatch/train.sh
```

## Caveats

- The project is named **subliminal-prompting**. Rename the folder (and references in the
  sbatch scripts under `cd ~/projects/...`) if you prefer something else.
- All sbatch scripts use `--partition=whartonstat` by default. Switch to `standby`
  for jobs >12h and rely on the auto-requeue pattern in `long_train.sh`.
- `src/train.py` is a minimal scaffold — replace `train_one_step` with your actual
  training loop. The checkpointing harness around it is production-shaped.
