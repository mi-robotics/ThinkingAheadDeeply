"""The base VLM: register the HF classes, and build its initial weights without the Prismatic framework.

The base model is Prism `prism-qwen25-extra-dinosiglip-224px+0_5b`: a DINOv2 ViT-L/14 (reg4) + SigLIP SO400M/14
fused vision backbone, a 3-layer GELU MLP projector, and Qwen2.5-0.5B. Its checkpoint holds the projector and the
full LLM; the two vision towers are the stock timm pretrained weights. `base_vlm_state_dict` assembles exactly the
state dict the Prismatic loader produced (`load(...).state_dict()` renamed to the HF module names), which the test
suite checks tensor for tensor.

RNG. The Prismatic loader ended by calling torch.manual_seed(2176) (the fused vision width) and initialising the
projector, so every training run's torch RNG was reset to the same stream at this point, whatever the training
seed. `base_vlm_state_dict(..., legacy_rng=True)` reproduces that end state exactly (used only by the equivalence
tests); training re-seeds with its own seed right after (recurrent_vla/train.py), so the seed now matters.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from recurrent_vla.vlm.configuration_prismatic import OpenVLAConfig
from recurrent_vla.vlm.modeling_prismatic import OpenVLAForActionPrediction
from recurrent_vla.vlm.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

DINO_TIMM_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_TIMM_ID = "vit_so400m_patch14_siglip_224"
PRISM_CHECKPOINT = "checkpoints/step-020792-epoch-01-loss=0.5268.pt"
IMAGE_SIZE = 224


def register_vlm_classes() -> None:
    """Make AutoConfig / AutoProcessor / AutoModelForVision2Seq resolve model_type "openvla" to this package."""
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def base_vlm_state_dict(vlm_path, legacy_rng: bool = False) -> dict:
    """Initial VLM weights in the HF OpenVLAForActionPrediction naming (loaded with strict=False:
    `action_queries` is not part of the base model and keeps its fresh initialisation)."""
    import timm

    raw = {}
    for prefix, timm_id in (("vision_backbone.featurizer.", DINO_TIMM_ID),
                            ("vision_backbone.fused_featurizer.", SIGLIP_TIMM_ID)):
        tower = timm.create_model(timm_id, pretrained=True, num_classes=0, img_size=IMAGE_SIZE)
        for k, v in tower.state_dict().items():
            # timm's LayerScale `gamma` is `scale_factor` in the HF model (HF overwrites params named gamma).
            raw[prefix + k.replace("gamma", "scale_factor")] = v
        del tower

    ckpt = torch.load(Path(vlm_path) / PRISM_CHECKPOINT, map_location="cpu")["model"]
    for i, k in enumerate(("fc1", "fc2", "fc3")):        # Prismatic's Sequential(Linear, GELU, Linear, GELU, Linear)
        for p in ("weight", "bias"):
            raw[f"projector.{k}.{p}"] = ckpt["projector"][f"projector.{2 * i}.{p}"]
    for k, v in ckpt["llm_backbone"].items():            # keys "llm.<...>"
        raw["language_model." + k[len("llm."):]] = v

    if legacy_rng:
        vision_dim = raw["vision_backbone.featurizer.norm.weight"].numel() + raw["vision_backbone.fused_featurizer.norm.weight"].numel()
        llm_dim = raw["projector.fc3.weight"].shape[0]
        torch.manual_seed(vision_dim)
        nn.Linear(vision_dim, 4 * vision_dim), nn.Linear(4 * vision_dim, llm_dim), nn.Linear(llm_dim, llm_dim)
    return raw
