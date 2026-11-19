"""Full training-state persistence: optimiser, scheduler, step, RNG, and provenance.

Why this exists
---------------
Myriad's continuation chain restarts the *same command* after each 48h segment. Twelve of the
24 final runs need more than one segment. Before this module, `save_training_checkpoint` wrote
only weights, so a continuation would have restarted at step 0 with a fresh AdamW -- producing
a loss curve that looks plausible and is wrong. The optimiser's second-moment estimates and the
LR schedule position are as much a part of the run as the weights.

What is restored exactly
------------------------
  * optimiser state (all of them -- the Muon path uses two)
  * LR scheduler position
  * gradient step counter
  * RNG: torch CPU, torch CUDA, numpy, python

What is NOT restored, and why
-----------------------------
The data position. RLDS/tf.data is an infinite, shuffled, interleaved stream; it exposes no
serialisable cursor, and `tf.data`'s own checkpointing does not survive the dataset being
rebuilt from a fresh graph in a new process. Pretending otherwise would be the expensive kind
of bug, so instead the stream is deliberately RE-SEEDED per segment (seed + start_step) in
finetune.py. Consequence, stated plainly: a resumed run draws a fresh sample of the same
distribution rather than continuing the identical sample order. For a shuffled 100k-step run
over a dataset that is revisited many times this changes nothing statistically, but it does
mean a resumed run is not bit-identical to an uninterrupted one. Runs that resume are marked
as such in their provenance file.

Layout, relative to the dir passed in (MYRIAD_CKPT_DIR when set, else <run_dir>/train_state):

    train_state/state.pt         the blob below, written atomically (tmp + os.replace)
    train_state/provenance.json  git hash, dirty flag, command, config, segment history
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

STATE_FILE = "state.pt"
PROVENANCE_FILE = "provenance.json"


# --- weight EMA (RDVLA_EMA, mem-hvla 2026-11-03) ------------------------------------------
class ParamEMA:
    """fp32 EMA of a list of parameters, with diffusion_policy's EMAModel schedule
    (inv_gamma 1, power 0.75, min 0, max 0.9999, update_after_step 0):
    decay = 1 - (1 + step)^-power with step = updates so far - 1, and 0 for the first two updates."""

    def __init__(self, params, power=0.75, max_value=0.9999):
        self.params = list(params)
        self.power, self.max_value = power, max_value
        self.shadow = [q.detach().float().clone() for q in self.params]
        self.num_updates = 0

    def decay(self):
        # EMAModel computes the decay from its counter BEFORE incrementing it; num_updates is that counter here.
        step = max(0, self.num_updates - 1)
        if step <= 0:
            return 0.0
        return min(1.0 - (1.0 + step) ** -self.power, self.max_value)

    @torch.no_grad()
    def update(self):
        d = self.decay()
        self.num_updates += 1
        for s, q in zip(self.shadow, self.params):
            s.mul_(d).add_(q.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to_params(self):
        for s, q in zip(self.shadow, self.params):
            q.copy_(s.to(q.dtype))

    def state_dict(self):
        return {"num_updates": self.num_updates, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        if len(sd["shadow"]) != len(self.shadow):
            raise RuntimeError(f"EMA mismatch: saved {len(sd['shadow'])} tensors, model has {len(self.shadow)}")
        for s, v in zip(self.shadow, sd["shadow"]):
            s.copy_(v.to(s.device))
        self.num_updates = int(sd["num_updates"])


# --- RNG ---------------------------------------------------------------------------------

def capture_rng(device_id=None):
    blob = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        blob["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return blob


def restore_rng(blob):
    """Restore RNG. Missing or mismatched entries are skipped loudly rather than silently."""
    if not blob:
        return
    try:
        torch.set_rng_state(blob["torch_cpu"].cpu().to(torch.uint8))
    except Exception as e:  # noqa: BLE001
        print(f"[resume] could not restore torch CPU RNG: {e}", flush=True)
    if "torch_cuda_all" in blob and torch.cuda.is_available():
        try:
            states = blob["torch_cuda_all"]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in states])
            else:
                print(f"[resume] CUDA device count changed "
                      f"({len(states)} saved vs {torch.cuda.device_count()} now); "
                      f"not restoring CUDA RNG", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[resume] could not restore CUDA RNG: {e}", flush=True)
    for key, setter in (("numpy", np.random.set_state), ("python", random.setstate)):
        if key in blob:
            try:
                setter(blob[key])
            except Exception as e:  # noqa: BLE001
                print(f"[resume] could not restore {key} RNG: {e}", flush=True)


# --- save / load -------------------------------------------------------------------------

def state_dir(run_dir) -> Path:
    """Where training state goes. MYRIAD_CKPT_DIR wins so the cluster's stall check -- which
    watches that dir for new writes between segments -- sees this file appear."""
    env = os.environ.get("MYRIAD_CKPT_DIR")
    return Path(env) if env else Path(run_dir) / "train_state"


def save(run_dir, *, step, checkpoint_dir, optimizers, scheduler, extra=None):
    """Write the full training state atomically. Returns the path written."""
    d = state_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": int(step),
        "checkpoint_dir": str(checkpoint_dir),
        "optimizers": [o.state_dict() for o in optimizers],
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng(),
        "saved_at": time.time(),
        "extra": extra or {},
    }
    final = d / STATE_FILE
    tmp = d / f"{STATE_FILE}.tmp.{os.getpid()}"
    torch.save(blob, tmp)
    os.replace(tmp, final)  # atomic within a filesystem: a killed job never leaves a torn file
    return final


def load(run_dir, map_location="cpu"):
    """Return the saved blob, or None when there is nothing to resume from."""
    p = state_dir(run_dir) / STATE_FILE
    if not p.exists():
        return None
    try:
        # weights_only=False: the blob holds numpy/python RNG tuples, not just tensors.
        return torch.load(p, map_location=map_location, weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"[resume] {p} exists but could not be read ({e}); starting fresh", flush=True)
        return None


def restore_into(blob, optimizers, scheduler, device_id=None):
    """Load optimiser/scheduler/RNG out of a blob. Returns the step to resume at."""
    saved = blob.get("optimizers") or []
    if len(saved) != len(optimizers):
        raise RuntimeError(
            f"resume mismatch: checkpoint has {len(saved)} optimiser(s), this run built "
            f"{len(optimizers)}. The optimiser configuration (e.g. use_muon) must match.")
    for opt, sd in zip(optimizers, saved):
        opt.load_state_dict(sd)
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    restore_rng(blob.get("rng"))
    return int(blob["step"])


# --- provenance --------------------------------------------------------------------------

def git_info(repo_root=None):
    """Commit hash and dirty flag, from git if possible and from the environment if not.

    On Myriad the project is rsynced WITHOUT .git (myr always excludes it), so `git rev-parse`
    there fails and every provenance file written on the cluster would silently record no
    commit at all -- the "save the git hash for exact recovery" requirement failing quietly,
    which is the worst way for it to fail. The submitting side therefore stamps
    RDVLA_GIT_COMMIT / RDVLA_GIT_DIRTY into the job environment, and those are used when git
    is unavailable. `source` says which path produced the answer, so a provenance file is
    never ambiguous about how much to trust its hash.
    """
    root = str(repo_root) if repo_root else str(Path(__file__).resolve().parents[2])

    def run(args):
        return subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10)

    err = None
    try:
        h = run(["git", "rev-parse", "HEAD"])
        if h.returncode == 0:
            status = run(["git", "status", "--porcelain"])
            branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            dirty_files = [l for l in status.stdout.splitlines() if l.strip()]
            return {
                "commit": h.stdout.strip(),
                "branch": branch.stdout.strip() or None,
                "dirty": bool(dirty_files),
                "dirty_files": dirty_files[:40],
                "repo_root": root,
                "source": "git",
            }
        err = h.stderr.strip()[:200]
    except Exception as e:  # noqa: BLE001
        err = str(e)[:200]

    env_commit = os.environ.get("RDVLA_GIT_COMMIT")
    if env_commit:
        return {
            "commit": env_commit,
            "branch": os.environ.get("RDVLA_GIT_BRANCH") or None,
            "dirty": os.environ.get("RDVLA_GIT_DIRTY") == "1",
            "repo_root": root,
            "source": "env (no .git here; stamped at submit time)",
            "git_error": err,
        }
    return {"commit": None, "error": err, "source": "unavailable"}


def _jsonable(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "__dict__"):
        return {str(k): _jsonable(x) for k, x in vars(v).items()}
    return str(v)


def write_provenance(target_dir, cfg, *, step=None, resumed_from=None, note=None):
    """Record exactly what produced this directory: commit, config, command, environment.

    Appended to, not overwritten: each segment of a continued run adds an entry, so the
    provenance file is the run's history rather than only its last moment.
    """
    d = Path(target_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / PROVENANCE_FILE

    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            existing = {}

    entry = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "step": step,
        "git": git_info(),
        "argv": list(__import__("sys").argv),
        "resumed_from": resumed_from,
        "note": note,
        "myriad": {k: os.environ.get(k) for k in
                   ("MYRIAD_RUN_ID", "MYRIAD_SEGMENT", "MYRIAD_CKPT_DIR", "MYRIAD_DEADLINE",
                    "JOB_ID") if os.environ.get(k)},
        "rdvla_env": {k: v for k, v in os.environ.items() if k.startswith("RDVLA_")},
        # Recorded ALWAYS, not only when the env var is set, so every checkpoint says which
        # action-block padding produced it. datasets.py used to fill the VLM's fixed 64-slot
        # action block with random.choices over the chunk (redrawn per load, all positions
        # supervised); since 2026-10-08 the default is a deterministic repeat. A provenance
        # entry with NO "action_pad_mode" key predates the change and is therefore
        # "legacy_random" -- that covers every checkpoint trained before 2026-10-08.
        "action_pad_mode": ("legacy_random"
                            if os.environ.get("RDVLA_LEGACY_ACTION_PAD") == "1"
                            else "deterministic"),
    }

    if not existing:
        existing = {"config": _jsonable(cfg), "segments": []}
    existing.setdefault("segments", []).append(entry)

    tmp = d / f"{PROVENANCE_FILE}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(existing, indent=1, default=str))
    os.replace(tmp, p)
    return p


# --- deadline ----------------------------------------------------------------------------

def deadline_passed(margin_s=0.0):
    """True when MYRIAD_DEADLINE (epoch seconds) is within margin_s of now.

    The cluster sends SIGTERM at h_rt minus --margin-min, then SIGKILL up to 5 min later.
    Watching the deadline directly lets the loop finish its micro-batch, save, and exit 99
    (the 'continue' code) rather than being killed mid-write.
    """
    raw = os.environ.get("MYRIAD_DEADLINE")
    if not raw:
        return False
    try:
        return time.time() >= float(raw) - margin_s
    except ValueError:
        return False
