"""Named dataset mixtures (LIBERO only). A dataset name that is not a mixture is loaded on its own."""

from typing import Dict, List, Tuple

OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    "libero_spatial_no_noops": [
        ("libero_spatial_no_noops", 1.0),
    ],
    "libero_object_no_noops": [
        ("libero_object_no_noops", 1.0),
    ],
    "libero_object_250hz_no_noops": [
        ("libero_object_250hz_no_noops", 1.0),
    ],
    "libero_goal_no_noops": [
        ("libero_goal_no_noops", 1.0),
    ],
    "libero_10_no_noops": [
        ("libero_10_no_noops", 1.0),
    ],
    "libero_4_task_suites_no_noops": [
        ("libero_spatial_no_noops", 1.0),
        ("libero_object_no_noops", 1.0),
        ("libero_goal_no_noops", 1.0),
        ("libero_10_no_noops", 1.0),
    ],

}
