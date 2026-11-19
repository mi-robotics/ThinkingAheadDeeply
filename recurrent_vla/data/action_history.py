"""Action history (RDVLA_ACTION_HISTORY / RDVLA_ACTION_GAP / RDVLA_ACTION_HIST_DROP), shared by training and eval.

Ported from the action-gap history study (RRD-VLA branch Exp-action-gap-history, local_setup/hist_common.py):
the policy additionally sees the actions of steps t-H .. t-1-G -- the last H steps with the G most recent withheld,
so with G > 0 it cannot simply copy a_{t-1} ("actgap": H = 10, G = 2). The window is in the normalised action space
of the training targets (arm by the q01/q99 bounds, gripper 1 = open / 0 = close) and is padded before the episode
start with the env's no-op action (no motion, gripper open). A zero-initialised MLP turns it into a term added to the
proprio token (models/hvla_context.py), so a fresh history arm starts exactly at the no-history model.

Training takes the window from the demonstration's actions (traj level, data/rlds/dataset.py); evaluation from the
actions the policy actually executed, gripper binarised at 0.5 (eval/run_libero_eval.py), as in the original study.
"""
import numpy as np

from recurrent_vla.constants import ACTION_GAP, ACTION_HISTORY

# get_libero_dummy_action: no motion, gripper -1 = open. In the dataset's transformed space (libero_dataset_transform
# clips the gripper to [0, 1] and inverts it) that gripper reads 1 = open.
NOOP_ACTION_TRANSFORMED = np.array([0, 0, 0, 0, 0, 0, 1], np.float64)


def window_len() -> int:
    """Rows of the history window the model sees: H - G."""
    return ACTION_HISTORY - ACTION_GAP


def normalize_actions(x, stats: dict, binarize_gripper: bool = False) -> np.ndarray:
    """Transformed-space actions (arm unnormalised, gripper in [0, 1]) -> the normalised target space, exactly as
    data/rlds/utils/data_utils.normalize_action_and_proprio does it for BOUNDS_Q99 (clip, mask, zero unused dims)."""
    x = np.asarray(x, np.float64)
    lo, hi = np.asarray(stats["q01"], np.float64), np.asarray(stats["q99"], np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, bool)), bool)
    n = np.where(mask, np.clip(2 * (x - lo) / (hi - lo + 1e-8) - 1, -1, 1), x)
    n = np.where(np.asarray(stats["min"]) == np.asarray(stats["max"]), 0.0, n)
    if binarize_gripper:
        n[..., -1] = (np.asarray(x)[..., -1] > 0.5).astype(np.float64)
    return n


def pad_action(stats: dict) -> np.ndarray:
    """The normalised no-op action that fills the window before the episode starts."""
    return normalize_actions(NOOP_ACTION_TRANSFORMED, stats).astype(np.float32)
