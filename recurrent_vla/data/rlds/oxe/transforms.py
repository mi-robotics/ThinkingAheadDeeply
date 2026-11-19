"""Per-dataset standardization transforms (LIBERO only).

Input: a dictionary of *batched* features (leading time dimension). Output: the same with "action" in EEF delta
+ gripper (1 = open, 0 = close) and "observation" carrying EEF_state (6) and gripper_state (2).
"""

from typing import Any, Dict

import tensorflow as tf

from recurrent_vla.data.rlds.utils.data_utils import invert_gripper_actions


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_object_250hz_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_4_task_suites_no_noops": libero_dataset_transform,
}
