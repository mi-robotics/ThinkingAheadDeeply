"""Policy loading and querying for evaluation: the VLM, proprio projector and HVLA head from a checkpoint dir, image
preprocessing matching training, and one policy query -> an action chunk."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from recurrent_vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType
from recurrent_vla.models.projectors import ProprioProjector
from recurrent_vla.models.recurrent_core import RecurrentConfigInternal
from recurrent_vla.vlm.base import register_vlm_classes

# TF only does image pre-processing here (center crop / resize); keep it off the GPU. This also matches training,
# where the RLDS pipeline hides the GPU from TF -- TF's GPU crop_and_resize differs from the CPU one in the last bit.
tf.config.set_visible_devices([], "GPU")

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    assert os.path.isdir(pretrained_checkpoint)
    checkpoint_files = [os.path.join(pretrained_checkpoint, f) for f in os.listdir(pretrained_checkpoint)
                        if file_pattern in f and "checkpoint" in f and f.endswith(".pt")]
    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in: {pretrained_checkpoint}"
    )
    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def get_vla(cfg: Any) -> torch.nn.Module:
    """The merged VLM from a checkpoint dir (model.safetensors + config.json), with its dataset statistics."""
    print("Instantiating pretrained VLA policy...")
    register_vlm_classes()
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False,
    )
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    vla = vla.to(DEVICE)
    stats = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats):
        with open(stats, "r") as f:
            vla.norm_stats = json.load(f)
    else:
        print("WARNING: No dataset_statistics.json found for checkpoint.")
    return vla


def get_processor(cfg: Any) -> AutoProcessor:
    register_vlm_classes()
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()
    proprio_projector.load_state_dict(load_component_state_dict(find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")))
    return proprio_projector


def get_action_head(cfg: Any, llm_dim: int):
    config_path = None
    config_files = list(Path(cfg.pretrained_checkpoint).glob("action_head_config--*.json"))
    if config_files:
        config_path = config_files[0]

    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            saved_cfg = json.load(f)

        head_type = saved_cfg.pop('_type', None)
        print(f"Auto-detected action head type: {head_type}")
        for key in ['prelude_vlm_layers', 'recurrent_vlm_layers', 'coda_vlm_layers']:
            if key in saved_cfg:
                saved_cfg[key] = tuple(saved_cfg[key])
        if head_type == "ActionHeadHVLA":
            from recurrent_vla.models.hvla_recurrent import ActionHeadHVLA as head_class
        elif head_type == "ActionHeadHVLAFlow":
            from recurrent_vla.models.hvla_flow import ActionHeadHVLAFlow as head_class
        else:
            raise ValueError(f"Unsupported action head type {head_type!r} in {config_path}: this codebase evaluates "
                             f"ActionHeadHVLA / ActionHeadHVLAFlow only (older heads: see the cvpr_2027_checkpoint branch)")
        action_head = head_class(hidden_dim=llm_dim, cfg=RecurrentConfigInternal(**saved_cfg))
    else:
        raise ValueError(
            f"No action_head_config JSON found in checkpoint: {cfg.pretrained_checkpoint}"
        )

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    state_dict = load_component_state_dict(find_checkpoint_file(cfg.pretrained_checkpoint, "action_head"))

    action_head.load_state_dict(state_dict)
    return action_head


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack([height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1)
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE))

    if expanded_dims:
        image = image[0]
    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, 0.9, 1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    assert (isinstance(image, np.ndarray) and len(image.shape) == 3
            and image.shape[-1] == 3 and image.dtype == np.uint8), \
        "Image must be numpy array with shape (H, W, 3) and dtype np.uint8"


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported normalization type")

    return np.clip(
        np.where(mask, 2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1, proprio),
        a_min=-1.0, a_max=1.0,
    )


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    processed_images = []
    for image in images:
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image).convert("RGB")
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)
        processed_images.append(pil_image)
    return processed_images


def get_vla_action(
    cfg: Any, vla: torch.nn.Module, processor: Any, obs: Dict[str, Any], task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
) -> List[np.ndarray]:
    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)

        prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n<|im_start|>assistant\n'

        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        # RDVLA_IMAGE_HISTORY: hand the rolling frame history to the HVLA S1 head (read in its build_context)
        _s1 = getattr(action_head, "model", None)
        if getattr(_s1, "image_history", 1) > 1 and "hist_images" in obs:
            _s1._hist_images = torch.from_numpy(np.asarray(obs["hist_images"]))[None].to(DEVICE)
        if getattr(_s1, "action_history", 0) and "action_hist" in obs:
            _s1._action_hist = torch.from_numpy(np.asarray(obs["action_hist"]))[None].to(DEVICE)

        convergence_strategy = getattr(cfg, 'recurrence_strategy', 'fixed')
        if convergence_strategy == 'fixed':
            convergence_strategy = None
        action, _, actual_iters, final_kl = vla.predict_action(
            **inputs, unnorm_key=cfg.unnorm_key, do_sample=False,
            proprio=proprio, proprio_projector=proprio_projector, action_head=action_head,
            num_iter=getattr(cfg, 'recurrent_num_iter', None),
            convergence_strategy=convergence_strategy,
            kl_thresh=getattr(cfg, 'recurrence_kl_thresh', 0.001),
            cos_thresh=getattr(cfg, 'recurrence_cos_thresh', 0.999),
            max_iter=getattr(cfg, 'recurrence_max_iter', 32),
        )

    actions = [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]
    return actions, actual_iters, final_kl
