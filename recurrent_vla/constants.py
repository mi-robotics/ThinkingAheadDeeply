import os
import sys
from typing import Optional
from enum import Enum

# Qwen2.5-0.5B token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 151386
STOP_INDEX = 2  # '</s>'
NUM_TOKENS = 64


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# LIBERO is the only platform this codebase trains and evaluates on.
ROBOT_PLATFORM = "LIBERO"
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
constants = LIBERO_CONSTANTS

# Opt-in override of the action chunk length (RDVLA_NUM_ACTIONS_CHUNK), default OFF = 8. Set it for both
# training and evaluation: modeling_prismatic.py's eval path reshapes to NUM_ACTIONS_CHUNK x ACTION_DIM, so a
# checkpoint evaluated under a different chunk would decode the wrong shape. Hence an env gate used by
# the one experiment that wants it; `action_chunk_len` in the head config must match.
_chunk_override = os.environ.get("RDVLA_NUM_ACTIONS_CHUNK")
if _chunk_override:
    constants = dict(constants)
    constants["NUM_ACTIONS_CHUNK"] = int(_chunk_override)
    print(f"[constants] NUM_ACTIONS_CHUNK overridden to {_chunk_override} "
          f"via RDVLA_NUM_ACTIONS_CHUNK", flush=True)

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Opt-in proprio history (2026-10-30), default 1 = off, so every existing run and checkpoint is unchanged. With
# RDVLA_PROPRIO_HISTORY=H the policy sees the last H proprio states (oldest first, padded at the start of an episode
# by repeating the first state), flattened into one H * PROPRIO_DIM vector: the proprio projector is an MLP over a
# flat vector, so only its input width changes. Training (traj_transforms.chunk_act_obs, finetune.py) and evaluation
# (run_libero_eval.py) must use the same H; scripts/train.sh records it in run.env and eval_libero.sh restores it.
PROPRIO_HISTORY = int(os.environ.get("RDVLA_PROPRIO_HISTORY", "1") or 1)

# Opt-in image history (2026-11-01), default 1 = off. With RDVLA_IMAGE_HISTORY=H the HVLA S1 encoder additionally sees
# the DINOv2 CLS token of each camera for the last H frames (t-H+1 .. t, oldest first, the first frame repeated at an
# episode start), next to the full patch tokens of the current frame -- after "Revisiting Open-Loop Action Chunking"
# (Zeng et al.). Data loading (datasets.py), the S1 context (hvla_context.py) and the LIBERO eval read it; set it for both
# training and evaluation (scripts/train.sh + eval_libero.sh do this via run.env). See hvla_context.py for RDVLA_HIST_VIT.
IMAGE_HISTORY = int(os.environ.get("RDVLA_IMAGE_HISTORY", "1") or 1)

# Opt-in action history (2026-11-05, from the action-gap study on RRD-VLA branch Exp-action-gap-history), default 0 =
# off. RDVLA_ACTION_HISTORY=H feeds the actions of steps t-H .. t-1-G (G = RDVLA_ACTION_GAP, the most recent steps
# withheld so a_{t-1} cannot be copied) through a zero-initialised MLP added to the proprio token. RDVLA_ACTION_HIST_DROP
# zeroes that term for this fraction of training samples ("actdrop"; eval always sees it). See data/action_history.py.
ACTION_HISTORY = int(os.environ.get("RDVLA_ACTION_HISTORY", "0") or 0)
ACTION_GAP = int(os.environ.get("RDVLA_ACTION_GAP", "0") or 0)
ACTION_HIST_DROP = float(os.environ.get("RDVLA_ACTION_HIST_DROP", "0") or 0)
if ACTION_HISTORY and not 0 <= ACTION_GAP < ACTION_HISTORY:
    raise ValueError(f"RDVLA_ACTION_GAP={ACTION_GAP} must be in [0, RDVLA_ACTION_HISTORY={ACTION_HISTORY})")

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
if IMAGE_HISTORY > 1:
    print(f"  IMAGE_HISTORY = {IMAGE_HISTORY} (RDVLA_IMAGE_HISTORY): S1 sees CLS tokens of the last {IMAGE_HISTORY} frames")
if ACTION_HISTORY:
    print(f"  ACTION_HISTORY = {ACTION_HISTORY}, gap {ACTION_GAP}, train dropout {ACTION_HIST_DROP}: actions t-{ACTION_HISTORY} .. t-{1 + ACTION_GAP}")
if PROPRIO_HISTORY > 1:
    print(f"  PROPRIO_HISTORY = {PROPRIO_HISTORY} (RDVLA_PROPRIO_HISTORY): proprio input is {PROPRIO_HISTORY * PROPRIO_DIM}-dim")
print("History / chunk switches: RDVLA_* environment variables, see recurrent_vla/constants.py")
