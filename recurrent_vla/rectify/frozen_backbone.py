"""Frozen-backbone utilities shared by the rectification scripts (rectify_recurrent.py, rectify_fast.py,
train_scratch_student.py): load a trained HVLA checkpoint's VLM, proprio projector and head, and build the
(pixel_values, s2_latent, proprio token, GT) context the recurrent head consumes.

Hidden-state extraction mirrors recurrent_vla/train.py:run_forward_pass. The LLM forward ignores proprio
(it only reaches the head through the projector), matching what inference feeds the head.
"""
import json
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

from recurrent_vla.vlm.base import register_vlm_classes
from recurrent_vla.models.recurrent_core import RecurrentConfigInternal
from recurrent_vla.models.projectors import ProprioProjector
from recurrent_vla.data.collator import PaddedCollatorForActionPrediction
from recurrent_vla.data.action_tokenizer import ActionTokenizer
from recurrent_vla.constants import IMAGE_HISTORY, PROPRIO_DIM, PROPRIO_HISTORY
from recurrent_vla.data.datasets import RLDSBatchTransform, RLDSDataset

class _NoLogits(torch.nn.Module):
    """The heads only need hidden states. The multimodal forward calls the LLM with labels=None and discards
    its logits, so skip the 151k-vocab projection (5.7 GB of float32 logits at batch 16)."""

    def forward(self, x):
        return x[..., :1]


def load_component(ckpt: Path, pattern: str):
    f = sorted(Path(ckpt).glob(pattern))[0]
    sd = torch.load(f, map_location="cpu", weights_only=True)
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, f.name


def load_frozen_backbone(ckpt: Path, dev):
    register_vlm_classes()
    vla = AutoModelForVision2Seq.from_pretrained(str(ckpt), torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
                                                 trust_remote_code=False).to(dev)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.eval().requires_grad_(False)
    vla.language_model.lm_head = _NoLogits()
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=False)
    proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM * PROPRIO_HISTORY)
    proprio_projector.load_state_dict(load_component(ckpt, "proprio_projector--*_checkpoint.pt")[0])
    proprio_projector = proprio_projector.to(torch.bfloat16).to(dev).eval().requires_grad_(False)
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    return vla, processor, proprio_projector, num_patches


def load_recurrent_head(ckpt: Path, llm_dim: int, dev):
    from recurrent_vla.constants import ACTION_HISTORY
    if ACTION_HISTORY:
        raise NotImplementedError("rectification does not support RDVLA_ACTION_HISTORY checkpoints yet")
    if IMAGE_HISTORY > 1:
        # The context is built from pixel_values + s2_latent only; the history frames would have to be threaded
        # through every cached batch and into the student's _hist_images as well. Not implemented -- refuse rather
        # than distil against a context with the history replaced by the current frame.
        raise NotImplementedError("rectification does not support RDVLA_IMAGE_HISTORY > 1 checkpoints yet")
    cfg = json.load(open(sorted(Path(ckpt).glob("action_head_config--*.json"))[0]))
    head_type = cfg.pop("_type", None)
    for k in ("prelude_vlm_layers", "recurrent_vlm_layers", "coda_vlm_layers"):
        cfg[k] = tuple(cfg[k])
    sd, fname = load_component(ckpt, "action_head--*_checkpoint.pt")
    if head_type == "ActionHeadHVLA":
        from recurrent_vla.models.hvla_recurrent import ActionHeadHVLA
        head = ActionHeadHVLA(hidden_dim=llm_dim, cfg=RecurrentConfigInternal(**cfg))
    else:
        raise ValueError(f"rectification needs a recurrent HVLA checkpoint (ActionHeadHVLA); {ckpt} has {head_type!r}")
    head.load_state_dict(sd)
    return head.to(torch.bfloat16).to(dev).eval().requires_grad_(False), fname


def make_loader(vla, processor, data_root_dir, dataset_name, batch_size, shuffle_buffer=20000, image_aug=True):
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                                   image_transform=processor.image_processor.apply_transform,
                                   use_wrist_image=True, use_proprio=True)
    dataset = RLDSDataset(data_root_dir, dataset_name, transform, resize_resolution=tuple(vla.config.image_sizes),
                          shuffle_buffer_size=shuffle_buffer, image_aug=image_aug)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id,
                                                 padding_side="right")
    return DataLoader(dataset, batch_size=batch_size, sampler=None, collate_fn=collator, num_workers=0)


def is_hvla_head(head):
    """Every head this codebase builds is HVLA: hooks take (pixel_values, s2_latent, p). Kept so the rectification
    scripts' IS_HVLA branches read unchanged; their non-HVLA branches are dead code."""
    return hasattr(head, "pool_prefix")


@torch.no_grad()
def extract_context(vla, proprio_projector, num_patches, batch, dev, head, dtype=torch.bfloat16):
    """One VLM forward -> (pixel_values, s2_latent, proprio token, normalized GT actions [B, 8, 7])."""
    use_cuda_amp = torch.device(dev).type == "cuda" and dtype == torch.bfloat16
    pixel_values = batch["pixel_values"].to(dtype).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
        out = vla(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
                  pixel_values=pixel_values, labels=batch["labels"].to(dev),
                  output_hidden_states=True, proprio=batch["proprio"], proprio_projector=proprio_projector)
    B = batch["input_ids"].shape[0]
    prop = proprio_projector(batch["proprio"].reshape(B, -1).to(dtype).to(dev)).unsqueeze(1)
    gt = batch["actions"].to(dev).to(dtype)
    s2 = head.pool_prefix(out.hidden_states, head.context_layer, batch["labels"], num_patches)
    return pixel_values, s2.to(dtype), prop, gt


def mirror_checkpoint(ckpt: Path, out: Path, skip_prefixes=("action_head--",)):
    """New checkpoint dir that run_libero_eval.py can load: small files copied, large ones symlinked."""
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(ckpt).iterdir():
        if f.name == ".cache" or any(f.name.startswith(s) for s in skip_prefixes):
            continue
        dst = out / f.name
        if dst.exists() or dst.is_symlink():
            continue
        if f.is_dir() or f.stat().st_size > 50e6:
            os.symlink(f.resolve(), dst)
        else:
            shutil.copy2(f, dst)
