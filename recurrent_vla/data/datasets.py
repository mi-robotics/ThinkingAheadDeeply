"""PyTorch IterableDataset over the RLDS (TFDS) pipeline, and the transform from an RLDS frame to the VLM's inputs.

Each sample: the Qwen2.5 chat prompt ("What action should the robot take to <instruction>?") followed by the
action-chunk tokens (padded to NUM_TOKENS by repeating the last), primary + wrist images through the VLM image
processor, the action chunk, proprio, and -- with RDVLA_IMAGE_HISTORY > 1 -- the raw history frames.
"""
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase

from recurrent_vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, IGNORE_INDEX, IMAGE_HISTORY, NUM_ACTIONS_CHUNK, NUM_TOKENS
from recurrent_vla.data.action_tokenizer import ActionTokenizer
from recurrent_vla.data.prompting import QwenPromptBuilder
from recurrent_vla.data.rlds import make_interleaved_dataset
from recurrent_vla.data.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def _get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _is_main_process() -> bool:
    return _get_dist_info()[0] == 0



@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Callable
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the collator / models."""
        hist_images = None
        if IMAGE_HISTORY > 1:
            # The RLDS window is IMAGE_HISTORY frames (oldest first). Keep the raw resized frames of both cameras
            # as the history, then reduce the batch to the usual window-1 layout so everything below is unchanged:
            # current observation = last window entry; actions = current + future (drop the window's past actions).
            obs = rlds_batch["observation"]
            hist_images = np.stack([obs["image_primary"], obs["image_wrist"]], axis=1)   # [H, 2, h, w, 3] uint8
            rlds_batch["observation"] = {k: (v[-1:] if hasattr(v, "shape") and len(v.shape) and v.shape[0] == IMAGE_HISTORY else v)
                                         for k, v in obs.items()}
            rlds_batch["action"] = rlds_batch["action"][IMAGE_HISTORY - 1:]
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        future_actions = rlds_batch["action"][1:]

        prompt_builder = QwenPromptBuilder("openvla")
        future_actions_string = self.action_tokenizer(future_actions, True)
        current_action_string = self.action_tokenizer(current_action, True)
        action_chunk_string = [current_action_string] + future_actions_string
        flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ''},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        if len(input_ids) >= 3:
            del input_ids[-3]
            del input_ids[-2]
            del input_ids[-1]

        if NUM_TOKENS < len(flattened_action_chunk_string):
            input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
        else:
            # Pad the action block to NUM_TOKENS by repeating its last token (deterministic; the positions stay
            # in the action mask so the stack the VLM forward assembles keeps NUM_TOKENS action positions).
            extended_array = [flattened_action_chunk_string[-1]] * (NUM_TOKENS - len(flattened_action_chunk_string))
            input_ids = input_ids + flattened_action_chunk_string + extended_array
        labels = list(input_ids)
        action_chunk_len = NUM_TOKENS

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Only the action tokens carry labels (they mark where the prompt prefix ends -- see pool_prefix).
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    all_wrist_pixels.append(self.image_transform(img_wrist))
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            return_dict["proprio"] = rlds_batch["observation"]["proprio"]
        if hist_images is not None:
            return_dict["hist_images"] = hist_images
        if "action_hist" in rlds_batch["observation"]:
            return_dict["action_hist"] = rlds_batch["observation"]["action_hist"][0]   # [H - G, A], normalised
        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        dataset_fraction: float = 1.0,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.dataset_fraction = dataset_fraction

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=IMAGE_HISTORY,                          # >1 only with RDVLA_IMAGE_HISTORY (see RLDSBatchTransform)
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                # For CPU-intensive ops (decoding, resizing, etc.); RDVLA_FRAME_THREADS overrides (e.g. on slow-CPU clusters)
                num_parallel_calls=int(os.environ.get("RDVLA_FRAME_THREADS", 16)),
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=int(os.environ.get("RDVLA_TRAJ_THREADS", len(mixture_spec))),
            traj_read_threads=int(os.environ.get("RDVLA_READ_THREADS", len(mixture_spec))),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

        # Apply dataset fraction if less than 100%
        if self.dataset_fraction < 1.0:
            num_samples = int(self.dataset_length * self.dataset_fraction)
            num_samples = max(1, num_samples)
            self.dataset = self.dataset.take(num_samples)
            self.dataset_length = num_samples
            if _is_main_process():
                print(f"Using {self.dataset_fraction*100:.1f}% of dataset: {num_samples} samples")

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        rank, world_size = _get_dist_info()
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        for idx, rlds_batch in enumerate(self.dataset.as_numpy_iterator()):
            if idx % num_shards != shard_id:
                continue
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        _, world_size = _get_dist_info()
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        num_shards = world_size * num_workers
        return math.ceil(self.dataset_length / num_shards)

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
