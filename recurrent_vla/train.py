"""Train an HVLA policy on LIBERO.

    python -m recurrent_vla.train --config recurrent_vla/configs/libero/hvla_recurrent_long.yaml [--key=value ...]
    (scripts/train.sh wraps this with torchrun, seeds, presets and run.env)

The VLM (S2) trains with LoRA; the HVLA head and the proprio projector train in full. Auto-resumes from the output
dir's train_state (optimiser, scheduler, RNG, EMA). History / architecture switches are RDVLA_* environment variables
(see recurrent_vla/constants.py and models/hvla_context.py).
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from recurrent_vla.vlm.base import base_vlm_state_dict, register_vlm_classes
from recurrent_vla.models.recurrent_core import RecurrentConfigInternal
from recurrent_vla.models.hvla_recurrent import ActionHeadHVLA
from recurrent_vla.models.hvla_flow import ActionHeadHVLAFlow
from recurrent_vla.config import ActionHeadType
from recurrent_vla import train_state as ts
from recurrent_vla.models.projectors import ProprioProjector
from recurrent_vla.data.collator import PaddedCollatorForActionPrediction
from recurrent_vla.data.action_tokenizer import ActionTokenizer
from recurrent_vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    PROPRIO_HISTORY,
)
from recurrent_vla.data.datasets import RLDSDataset, RLDSBatchTransform
from recurrent_vla.data.rlds.utils.data_utils import save_dataset_statistics

def _prefetch_batches(loader, depth):
    """Yield batches from `loader`, producing up to `depth` ahead in a background thread (RDVLA_PREFETCH).

    The RLDS loader runs with num_workers=0, so the per-sample PIL/tokenizer transforms in RLDSBatchTransform run
    in the training process; on slow-CPU nodes they serialise with GPU compute. A thread overlaps the two.
    When the consumer stops (loop exit / generator close), the worker is told to stop and joined, so the TF
    iterator is not torn down mid-call at interpreter exit (that aborts the process with SIGABRT).
    """
    import queue
    import threading

    q, done, stop = queue.Queue(maxsize=depth), object(), threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def work():
        try:
            for b in loader:
                if not put(b):
                    return
        except BaseException as e:  # surfaced in the consumer
            put(e)
            return
        put(done)

    worker = threading.Thread(target=work, daemon=True, name="rdvla-prefetch")
    worker.start()
    try:
        while True:
            b = q.get()
            if b is done:
                return
            if isinstance(b, BaseException):
                raise b
            yield b
    finally:
        stop.set()
        worker.join(timeout=120)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Test-only (tests/test_equivalence.py): reproduce the pre-refactor torch RNG stream, in which the Prismatic loader
# reset the generator to a fixed seed after the training seed was set (so the seed did not reach model init).
LEGACY_RNG = os.environ.get("RVLA_LEGACY_RNG") == "1"


class _SkipVocabProjection(nn.Module):
    """Stand-in for the LLM's lm_head during action-head training: returns a width-1 slice instead of vocab logits."""

    def __init__(self):
        super().__init__()
        self._reported = False

    def forward(self, hidden_states):
        if not self._reported:
            print(f"[lm_head skipped] hidden {tuple(hidden_states.shape)} -> logits placeholder {tuple(hidden_states[..., :1].shape)}", flush=True)
            self._reported = True
        return hidden_states[..., :1]


def remove_ddp_in_checkpoint(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg):
    if hasattr(cfg, 'run_id_override') and cfg.run_id_override is not None:
        return cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.config_file_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
        return run_id
    else:
        run_id = f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if hasattr(cfg, 'run_id_note') and cfg.run_id_note:
            run_id += f"--{cfg.run_id_note}"
        return run_id


def load_checkpoint(module_name, path, step, device="cpu", tag=None):
    # `tag` is the filename stem actually written by save_training_checkpoint. It is normally
    # "<step>_checkpoint.pt", but with save_latest_checkpoint_only it is "latest_checkpoint.pt"
    # and the step no longer appears in the name -- so an auto-resume must pass the tag it
    # recorded rather than reconstructing it from the step.
    suffix = tag or f"{step}_checkpoint.pt"
    checkpoint_path = os.path.join(path, f"{module_name}--{suffix}")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


class _FrozenModuleShim(torch.nn.Module):
    """Stands in for DDP around a fully-frozen module.

    DDP raises if a module has no parameter requiring grad, but the rest of finetune.py
    reaches through `.module` in nine places. This gives them that attribute, forwards calls,
    and delegates anything else (`.train()`, `.eval()`, `.parameters()`, ...) to the wrapped
    module, so a frozen S2 needs no special-casing anywhere else.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def wrap_ddp(module, device_id, find_unused=False):
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module, name):
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(module_class, module_name, cfg, device_id, module_args, to_bf16=False, find_unused_params=False):
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step,
                                     tag=getattr(cfg, "resume_ckpt_tag", None))
        module.load_state_dict(state_dict)
        print('Loaded checkpoint!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def _head(action_head):
    return action_head.module if hasattr(action_head, "module") else action_head


def _is_flow(head):
    """The flow head trains through `training_loss` (the flow MSE), never L1 against `predict_action`:
    sampling is a no-grad Euler integration and carries no gradient to the velocity field."""
    return isinstance(head, ActionHeadHVLAFlow)


def _s2_latent(head, raw_hidden_states, labels, num_patches):
    """The S2 latent: the VLM prefix (image patches + instruction tokens) mean-pooled at head.context_layer."""
    return head.pool_prefix(raw_hidden_states, head.context_layer, labels, num_patches)


def _set_history(head, batch):
    """Hand the batch's history inputs to the S1 context: frames (RDVLA_IMAGE_HISTORY, read in build_context) and
    actions (RDVLA_ACTION_HISTORY, read in proprio_token)."""
    M = getattr(head, "model", None)
    if getattr(M, "image_history", 1) > 1:
        M._hist_images = batch.get("hist_images")
    if getattr(M, "action_history", 0):
        M._action_hist = batch.get("action_hist")


def _supports_recurrent_probe(head):
    """The health/depth probes need the recurrent hooks; the flow head has none (flow_eval covers it)."""
    M = getattr(head, "model", None)
    return M is not None and all(
        hasattr(M, a) for a in
        ("build_prelude", "init_state", "adapter", "_run_one_iteration", "_get_output"))


def _vla_forward(vla, batch, device_id, proprio_projector, use_proprio=True):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
        )


@torch.no_grad()
def recurrence_health(action_head, pixel_values, s2_latent, proprio, proprio_projector, k_hi=12):
    """Is the recurrent head still using its state? Prelude-output RMS, the recurrent state's share of the adapter's
    input, and how much the decoded actions change between 1 and k_hi iterations from the same start noise."""
    head = _head(action_head)
    M = head.model
    B = pixel_values.shape[0]
    dev = s2_latent.device
    p = M.proprio_token(proprio_projector(proprio.reshape(B, -1).to(torch.bfloat16).to(dev)).unsqueeze(1))
    ctx_a = pixel_values.to(torch.bfloat16).to(dev)
    x = M.build_prelude(ctx_a, s2_latent, p)
    s = M.init_state(B, dev, x.dtype)
    D = s.shape[-1]
    W = M.adapter.weight.float()
    c_state = (s.float() @ W[:, :D].T).norm(dim=-1).mean()
    c_prelude = (x.float() @ W[:, D:].T).norm(dim=-1).mean()
    out_1 = None
    for k in range(1, k_hi + 1):
        s = M._run_one_iteration(s, x, ctx_a, s2_latent, p)
        if k == 1:
            out_1 = M._get_output(s, ctx_a, s2_latent, p)
    out_hi = M._get_output(s, ctx_a, s2_latent, p)
    return {
        "prelude_rms": x.float().pow(2).mean().sqrt().item(),
        "state_share": (c_state / (c_state + c_prelude)).item(),
        f"action_diff_k1_k{k_hi}": (out_1.float() - out_hi.float()).abs().mean().item(),
        # Spectral norms of the adapter's state and prelude blocks (init ~1.24).
        "adapter_sigma_state": torch.linalg.matrix_norm(W[:, :D], ord=2).item(),
        "adapter_sigma_prelude": torch.linalg.matrix_norm(W[:, D:], ord=2).item(),
    }


DEPTH_EVAL_ITERS = (1, 2, 4, 8, 12, 32)


@torch.no_grad()
def depth_eval(vla, action_head, proprio_projector, batches, device_id, num_patches, seed=1234):
    """Held-out action L1 after k recurrent iterations, on fixed validation batches and a fixed start noise, so
    evaluations at different training steps are directly comparable. Restores the RNG state and train mode."""
    head = _head(action_head)
    M = head.model
    was_training = vla.training
    vla.eval()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device_id)
    torch.manual_seed(seed)
    l1 = {k: 0.0 for k in DEPTH_EVAL_ITERS}
    diff_k1_k12 = 0.0
    for batch in batches:
        _set_history(head, batch)
        output = _vla_forward(vla, batch, device_id, proprio_projector)
        B = batch["input_ids"].shape[0]
        gt = batch["actions"].to(device_id).float()
        p = M.proprio_token(proprio_projector(batch["proprio"].reshape(B, -1).to(torch.bfloat16).to(device_id)).unsqueeze(1))
        ctx_a = batch["pixel_values"].to(torch.bfloat16).to(device_id)
        ctx_b = _s2_latent(head, output.hidden_states, batch["labels"], num_patches)
        x = M.build_prelude(ctx_a, ctx_b, p)
        s = M.init_state(B, ctx_a.device, x.dtype)
        outs = {}
        for k in range(1, max(DEPTH_EVAL_ITERS) + 1):
            s = M._run_one_iteration(s, x, ctx_a, ctx_b, p)
            if k in l1:
                outs[k] = M._get_output(s, ctx_a, ctx_b, p).float()
                l1[k] += (outs[k] - gt).abs().mean().item() / len(batches)
        diff_k1_k12 += (outs[1] - outs[12]).abs().mean().item() / len(batches)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device_id)
    if was_training:
        vla.train()
    result = {f"L1_k{k}": v for k, v in l1.items()}
    result["gain_k1_to_k12_pct"] = 100.0 * (l1[1] - l1[12]) / l1[1]
    result["gain_k1_to_k4_pct"] = 100.0 * (l1[1] - l1[4]) / l1[1]
    result["gain_k12_to_k32_pct"] = 100.0 * (l1[12] - l1[32]) / l1[12]
    result["action_diff_k1_k12"] = diff_k1_k12
    return result


FLOW_EVAL_NFES = (1, 2, 4, 8, 10)


@torch.no_grad()
def flow_eval(vla, action_head, proprio_projector, batches, device_id, num_patches, seed=1234):
    """The flow analogue of depth_eval: held-out action L1 after n Euler steps (one NFE ~ one recurrent
    iteration), on the same fixed batches and start noise."""
    head = _head(action_head)
    M = head.model
    was_training = vla.training
    vla.eval()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device_id)
    torch.manual_seed(seed)
    l1 = {n: 0.0 for n in FLOW_EVAL_NFES}
    for batch in batches:
        _set_history(head, batch)
        output = _vla_forward(vla, batch, device_id, proprio_projector)
        B = batch["input_ids"].shape[0]
        gt = batch["actions"].to(device_id).float()
        p = M.proprio_token(proprio_projector(batch["proprio"].reshape(B, -1).to(torch.bfloat16).to(device_id)).unsqueeze(1))
        s2 = _s2_latent(head, output.hidden_states, batch["labels"], num_patches)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for n in FLOW_EVAL_NFES:
                pred = M.sample(batch["pixel_values"].to(torch.bfloat16).to(device_id), s2, p, n)
                l1[n] += (pred.float() - gt).abs().mean().item() / len(batches)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device_id)
    if was_training:
        vla.train()
    result = {f"L1_nfe{n}": v for n, v in l1.items()}
    result["gain_nfe1_to_nfe10_pct"] = 100.0 * (l1[1] - l1[10]) / l1[1]
    result["gain_nfe1_to_nfe4_pct"] = 100.0 * (l1[1] - l1[4]) / l1[1]
    return result


def run_forward_pass(vla, action_head, proprio_projector, batch, action_tokenizer, device_id,
                     use_proprio, num_patches, cfg=None, health=False):
    head_mod = _head(action_head)
    _set_history(head_mod, batch)
    metrics = {}
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    output: CausalLMOutputWithPast = _vla_forward(vla, batch, device_id, proprio_projector, use_proprio)
    # S1 consumes pixels plus ONE pooled latent from S2 -- never the VLM token stack.
    s2_latent = _s2_latent(head_mod, output.hidden_states, batch["labels"], num_patches)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)

    if _is_flow(head_mod):
        # predict_action integrates under no_grad, so an L1 wired to it would backpropagate nothing.
        loss = head_mod.training_loss(
            ground_truth_actions, pixel_values, s2_latent,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
        )
        # No per-timestep L1 for a velocity field; flow_eval reports sampled L1 across NFE.
        metrics["loss_value"] = loss.item()
        return loss, metrics

    result = head_mod.predict_action(
        pixel_values, s2_latent,
        proprio=batch["proprio"] if use_proprio else None,
        proprio_projector=proprio_projector if use_proprio else None,
        phase=cfg.phase,
    )

    predicted_actions = result

    # Deep supervision: predicted_actions may be a list of predictions
    if isinstance(predicted_actions, list):
        n = len(predicted_actions)
        weights = [(i + 1) for i in range(n)]
        total_weight = sum(weights)

        losses = [torch.nn.L1Loss()(pred, ground_truth_actions) for pred in predicted_actions]
        loss = sum(w * l for w, l in zip(weights, losses)) / total_weight

        final_pred = predicted_actions[-1]
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_l1_loss": torch.nn.L1Loss()(final_pred[:, 0], ground_truth_actions[:, 0]).item(),
            "next_actions_l1_loss": torch.nn.L1Loss()(final_pred[:, 1:], ground_truth_actions[:, 1:]).item(),
            "deep_supervision_n": len(predicted_actions),
            "iter_0_loss": losses[0].item() if len(losses) > 0 else 0.0,
            "iter_final_loss": losses[-1].item() if len(losses) > 0 else 0.0,
        })
    else:
        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_l1_loss": torch.nn.L1Loss()(predicted_actions[:, 0], ground_truth_actions[:, 0]).item(),
            "next_actions_l1_loss": torch.nn.L1Loss()(predicted_actions[:, 1:], ground_truth_actions[:, 1:]).item(),
        })

    if health and use_proprio and _supports_recurrent_probe(head_mod):
        metrics["health"] = recurrence_health(action_head, pixel_values, s2_latent.detach(), batch["proprio"],
                                              proprio_projector)

    return loss, metrics


def save_training_checkpoint(cfg, run_dir, log_step, vla, processor, proprio_projector,
                             action_head, train_dataset, distributed_state, new_state_dict):
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        # Every save dir records the commit and the resolved config it came from, so a
        # checkpoint can be traced back to exact code without relying on the run directory.
        ts.write_provenance(checkpoint_dir, cfg, step=log_step, note="checkpoint")
        print(f"Saving Model Checkpoint for Step {log_step}")

    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        # Without LoRA there is no adapter: vla is the plain model, so saving it into
        # `lora_adapter/` leaves the checkpoint ROOT with no config.json and no weights.
        # Everything downstream -- frozen_backbone.load_frozen_backbone (rectification, flow
        # baseline), run_suite.sh's model.safetensors guard, and auto-resume, which refuses a
        # checkpoint dir with no config.json -- does from_pretrained(checkpoint_dir) and fails.
        # The merge branch below only runs when use_lora is set, so nothing else fixes it up.
        if not cfg.use_lora:
            vla.module.save_pretrained(checkpoint_dir)
        else:
            vla.module.save_pretrained(adapter_dir)

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if action_head is not None:
            action_head_unwrapped = getattr(action_head, 'module', action_head)
            torch.save(action_head_unwrapped.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
            if hasattr(action_head_unwrapped, 'cfg'):
                import json
                config_path = checkpoint_dir / f"action_head_config--{checkpoint_name_suffix.replace('.pt', '.json')}"
                cfg_dict = {k: list(v) if isinstance(v, tuple) else v for k, v in action_head_unwrapped.cfg.__dict__.items()}
                cfg_dict['_type'] = type(action_head_unwrapped).__name__
                with open(config_path, 'w') as f:
                    json.dump(cfg_dict, f, indent=2)

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        config = AutoConfig.from_pretrained(os.path.join(cfg.config_file_path, "config.json"))
        base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
        new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
        base_vla.load_state_dict(new_state_dict, strict=False)

        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        dist.barrier()

    # The caller records both so auto-resume can find the weights again: the directory, and
    # the filename stem, which is "latest_checkpoint.pt" rather than "<step>_checkpoint.pt"
    # when save_latest_checkpoint_only is set.
    return checkpoint_dir, checkpoint_name_suffix


def run_validation(
    vla, action_head, proprio_projector, val_dataloader, action_tokenizer,
    device_id, cfg, num_patches, log_step, distributed_state, val_time_limit,
):
    val_start_time = time.time()
    vla.eval()
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla, action_head=action_head, proprio_projector=proprio_projector,
                batch=batch, action_tokenizer=action_tokenizer, device_id=device_id,
                use_proprio=cfg.use_proprio,
                num_patches=num_patches, cfg=cfg,
            )
            all_val_metrics.append(metrics)
            if time.time() - val_start_time > val_time_limit:
                break

    if not all_val_metrics:
        vla.train()
        return

    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [m[metric_name] for m in all_val_metrics if metric_name in m]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    if distributed_state.is_main_process:
        if cfg.use_wandb:
            wandb.log({f"VLA Val/{k}": v for k, v in avg_val_metrics.items()}, step=log_step)
        else:
            print(f"[val step {log_step}] " + ", ".join(f"{k}={v:.6f}" for k, v in avg_val_metrics.items()))

    vla.train()


def _autoresume(cfg, run_dir):
    """Point cfg at the last checkpoint if a train_state blob exists. Returns (step, blob).

    Myriad restarts the *same command* for each continuation segment, so resume has to be
    automatic and driven by what is on disk -- there is no opportunity to pass a different
    flag on the second segment. Set RDVLA_NO_AUTORESUME=1 to force a fresh start.
    """
    if os.environ.get("RDVLA_NO_AUTORESUME") == "1":
        print("[resume] RDVLA_NO_AUTORESUME=1; starting fresh", flush=True)
        return 0, None
    # Weights-only resume from an explicit checkpoint: restores model weights and the step
    # counter but starts a FRESH optimiser. For re-finishing runs whose matching optimiser state
    # no longer exists (train_state keeps only the latest blob). The run's provenance records it.
    wdir = os.environ.get("RDVLA_RESUME_WEIGHTS_FROM")
    if wdir and ts.load(run_dir) is not None:
        # A train_state blob means this job already saved its own state (a later segment of the
        # re-finish): continue from that, never jump back to the explicit weights.
        print(f"[resume] ignoring RDVLA_RESUME_WEIGHTS_FROM: {run_dir} has its own train_state", flush=True)
        wdir = None
    if wdir:
        wstep = int(os.environ["RDVLA_RESUME_WEIGHTS_STEP"])
        if not os.path.exists(os.path.join(wdir, "config.json")):
            raise RuntimeError(f"RDVLA_RESUME_WEIGHTS_FROM={wdir} has no config.json")
        cfg.resume = True
        cfg.resume_step = wstep
        cfg.resum_vla_path = wdir
        cfg.config_file_path = wdir
        cfg.resume_ckpt_tag = os.environ.get("RDVLA_RESUME_WEIGHTS_TAG")
        cfg.resume_weights_only = True
        print(f"[resume] WEIGHTS-ONLY from {wdir} at step {wstep}; fresh optimiser", flush=True)
        return wstep, None
    blob = ts.load(run_dir)
    if blob is None:
        return 0, None
    ckpt, step = blob.get("checkpoint_dir"), int(blob.get("step", 0))
    if not ckpt or not os.path.isdir(ckpt):
        print(f"[resume] train_state names {ckpt!r}, which is not a directory; starting fresh",
              flush=True)
        return 0, None
    if not os.path.exists(os.path.join(ckpt, "config.json")):
        # Without a merged model in the checkpoint dir there is nothing to rebuild the VLM
        # from. This happens when merge_lora_during_training is off, or on the no-LoRA path.
        print(f"[resume] {ckpt} has no config.json (was merge_lora_during_training off?); "
              f"cannot resume weights, starting fresh", flush=True)
        return 0, None
    cfg.resume = True
    cfg.resume_step = step
    cfg.resum_vla_path = ckpt
    cfg.config_file_path = ckpt
    cfg.resume_ckpt_tag = (blob.get("extra") or {}).get("checkpoint_tag")
    print(f"[resume] continuing from step {step}; weights from {ckpt}", flush=True)
    return step, blob


def finetune(cfg):
    global RAW_STATE_DICT

    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning on `{cfg.dataset_name}`")

    # run_id must be derived BEFORE auto-resume rewrites config_file_path, or a continuation
    # would compute a different run_id than the segment it is continuing and write elsewhere.
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    resume_step, resume_blob = _autoresume(cfg, run_dir)

    # Seed. TrainConfig.seed existed but was never forwarded to the training path, so every
    # run before this was effectively unseeded. Offset by the resume step so a continuation
    # does not replay the first segment's augmentation and S_0 noise stream (the RLDS cursor
    # itself is not restorable -- see train_state.py).
    seed = int(getattr(cfg, "seed", 42)) + resume_step
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[seed] base={getattr(cfg, 'seed', 42)} resume_step={resume_step} -> {seed}", flush=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        ts.write_provenance(run_dir, cfg, step=resume_step,
                            resumed_from=(resume_blob or {}).get("checkpoint_dir"),
                            note="segment start")

    if distributed_state.is_main_process:
        if cfg.use_wandb:
            # Record the full training config plus the RDVLA_* diagnostics settings (rectified-recurrence project).
            try:
                wandb_config = {k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v) for k, v in vars(cfg).items()}
            except TypeError:
                wandb_config = {}
            wandb_config.update({k: v for k, v in os.environ.items() if k.startswith("RDVLA_")})
            wandb_kwargs = dict(project=cfg.wandb_project, name=os.environ.get("RDVLA_WANDB_NAME") or f"ft+{run_id}",
                                mode=os.environ.get("WANDB_MODE", "online"), config=wandb_config)
            if cfg.wandb_entity:
                wandb_kwargs["entity"] = cfg.wandb_entity
            try:
                wandb.init(**wandb_kwargs)
            except Exception as e:
                # A telemetry failure must never kill a multi-day run. wandb spawns a local
                # service process even under WANDB_MODE=offline; when that service fails to
                # come up, init() raises after a 90s timeout. Three 100k-step cells died this
                # way on 2026-09-30 and held their GPUs for hours. metrics.jsonl is written
                # independently and is the authoritative record, so drop wandb and continue.
                print(f"[wandb] init failed ({type(e).__name__}: {e}); continuing WITHOUT wandb",
                      flush=True)
                try:
                    cfg.use_wandb = False
                except Exception:
                    object.__setattr__(cfg, "use_wandb", False)

    print(f"NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}, ACTION_DIM: {ACTION_DIM}")

    register_vlm_classes()
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=False)

    # The base VLM's weights (stock timm vision towers + the Prism checkpoint's projector and LLM), loaded into the HF
    # model built from config.json. strict=False: `action_queries` is not in the base model and stays freshly drawn.
    RAW_STATE_DICT = base_vlm_state_dict(cfg.vlm_path, legacy_rng=LEGACY_RNG)
    if not LEGACY_RNG:
        # Seed again here so model initialisation (action queries, proprio projector, head, S_0 noise) follows the
        # training seed; building the base weights must not decide it.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    config = AutoConfig.from_pretrained(os.path.join(cfg.config_file_path, "config.json"))
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)
    vla.load_state_dict(RAW_STATE_DICT, strict=False)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # The action head trains on hidden states only: the multimodal forward calls the LLM with labels=None and the loss
    # is L1 on the head, so the 151k-vocab logits (upcast to float32, ~2.8 GB at batch 8) are never used. Replace the
    # projection module (overriding forward on the instance did not take effect). Qwen2.5-0.5B ties lm_head to the
    # input embeddings (text_config.tie_word_embeddings=True), so checkpoints re-tie it on load.
    vla.language_model.lm_head = _SkipVocabProjection()

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        if getattr(cfg, "resume", False):
            # RESUME BUG FIX (2026-10-06). On the minivlm path `vla` was just rebuilt from the
            # BASE Prism VLM (cfg.vlm_path); config_file_path only supplied config.json. The old
            # code then attached a FRESH LoRA, silently discarding all VLM fine-tuning at every
            # resume (held-out L1 at the same step: 0.191 -> 0.306). Re-attach the checkpoint's
            # trained adapter instead, and restore action_queries, which is trained but lives
            # outside the adapter (only the merged model.safetensors has it).
            adapter_dir = os.path.join(cfg.resum_vla_path, "lora_adapter")
            if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
                raise RuntimeError(f"[resume] {adapter_dir} has no adapter_config.json -- refusing to "
                                   f"resume onto a fresh LoRA (that is the bug this guards against)")
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
            from safetensors import safe_open
            merged = os.path.join(cfg.resum_vla_path, "model.safetensors")
            with safe_open(merged, framework="pt", device="cpu") as f:
                aq = f.get_tensor("action_queries.weight")
            tgt = vla.base_model.model.action_queries.weight
            with torch.no_grad():
                tgt.copy_(aq.to(tgt.device, tgt.dtype))
            print(f"[resume] restored trained LoRA adapter + action_queries from {cfg.resum_vla_path}",
                  flush=True)
        else:
            vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()
    else:
        # No LoRA => nothing has frozen the base model. PEFT does that for you on the LoRA
        # path, so this branch previously left the ENTIRE VLM trainable: the first HVLA smoke
        # reported 1,387,809,741 trainable params instead of the intended 134.4M (the S1 head
        # alone). For the HVLA split S2 must be a fixed scene encoder -- if it keeps training,
        # the pooled latent is non-stationary and S1 cannot cache it across steps, which is
        # the entire point of the split. Freeze explicitly.
        for param in vla.parameters():
            param.requires_grad = False
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        n_train = sum(p.numel() for p in vla.parameters() if p.requires_grad)
        print(f"[freeze] no-LoRA path: VLM frozen, {n_train:,} trainable params left in the VLM")

    # With use_lora=false (the HVLA split, where S2 is a frozen scene encoder) the VLM has NO
    # trainable parameters, and DDP refuses such a module outright:
    #   "DistributedDataParallel is not needed when a module doesn't have any parameter that
    #    requires a gradient."
    # Nine call sites reach through `vla.module`, so we cannot simply leave it unwrapped --
    # hence a shim that provides `.module` and forwards everything else. Only used when the
    # VLM is fully frozen; the LoRA path below is unchanged.
    if any(p.requires_grad for p in vla.parameters()):
        vla = wrap_ddp(vla, device_id, find_unused=True)
    else:
        print("[ddp] VLM has no trainable parameters (frozen S2) -- using a no-op wrapper")
        vla = _FrozenModuleShim(vla)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            # RDVLA_PROPRIO_HISTORY (constants.py): the projector reads H stacked states; H = 1 is the old width
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM * PROPRIO_HISTORY},
            to_bf16=True,
        )

    llm_dim = vla.module.llm_dim
    action_head = None
    action_head_type = getattr(cfg, 'action_head', None)
    if action_head_type is not None and hasattr(action_head_type, 'type'):
        rec_cfg = action_head_type.recurrent
        internal_cfg = RecurrentConfigInternal(
            hidden_dim=rec_cfg.hidden_dim,
            num_heads=rec_cfg.num_heads,
            prelude_vlm_layers=tuple(rec_cfg.prelude_vlm_layers),
            recurrent_vlm_layers=tuple(rec_cfg.recurrent_vlm_layers),
            coda_vlm_layers=tuple(rec_cfg.coda_vlm_layers),
            action_chunk_len=rec_cfg.action_chunk_len,
            action_dim=rec_cfg.action_dim,
            mean_recurrence=rec_cfg.mean_recurrence,
            backprop_depth=rec_cfg.backprop_depth,
            random_iterations=rec_cfg.random_iterations,
            init_std=rec_cfg.init_std,
            rms_norm_eps=rec_cfg.rms_norm_eps,
            rope_base=rec_cfg.rope_base,
            prelude_out_norm=getattr(rec_cfg, "prelude_out_norm", False),
            deep_supervision=getattr(rec_cfg, "deep_supervision", False),
            state_init_std=getattr(rec_cfg, "state_init_std", 0.0),
        )
        # ActionHeadType is a str-Enum; on py3.10 str(Enum.MEMBER) gives the member NAME, so read .value.
        _ht = getattr(action_head_type, "type", "hvla")
        head_type = str(getattr(_ht, "value", _ht)).lower()
        if head_type not in ("hvla", "hvla_flow"):
            raise ValueError(f"action_head.type={head_type!r}: expected hvla | hvla_flow")
        # Copied under hasattr(rec_cfg, f): a key RecurrentConfig (configs/base.py) does not declare is
        # dropped from the YAML in silence, so every field read here must exist there.
        for _f in ("s1_encoder_layers", "s1_context_layer", "s1_vit_model", "s1_vit_pretrained",
                   "s1_num_images", "s1_s2_refresh_every", "s1_freeze_vit",
                   "flow_num_layers", "flow_num_inference_steps"):
            if hasattr(rec_cfg, _f):
                setattr(internal_cfg, _f, getattr(rec_cfg, _f))
        head_class = ActionHeadHVLAFlow if head_type == "hvla_flow" else ActionHeadHVLA
        print(f"[action head] {head_class.__name__}: S1 = {internal_cfg.s1_vit_model} over "
              f"{internal_cfg.s1_num_images} images (frozen: {bool(internal_cfg.s1_freeze_vit)}), "
              f"image history {os.environ.get('RDVLA_IMAGE_HISTORY', '1')} "
              f"({os.environ.get('RDVLA_HIST_VIT', 'shared')}), proprio history "
              f"{PROPRIO_HISTORY}; S2 = VLM prefix mean-pooled at layer {internal_cfg.s1_context_layer}"
              + (f"; flow: {internal_cfg.flow_num_layers} layers, {internal_cfg.flow_num_inference_steps} NFE"
                 if head_type == "hvla_flow" else ""))

        action_head = init_module(
            head_class, "action_head", cfg, device_id,
            {"hidden_dim": llm_dim, "cfg": internal_cfg},
            to_bf16=True,
        )

    if action_head is None:
        raise ValueError("action_head config is required")

    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()

    vla_params = [param for param in vla.parameters() if param.requires_grad]
    action_head_params = [param for param in action_head.parameters() if param.requires_grad] if action_head else []
    proprio_params = [param for param in proprio_projector.parameters() if param.requires_grad] if cfg.use_proprio else []

    total_params = sum(p.numel() for p in vla_params + action_head_params + proprio_params)
    print(f"# total trainable params: {total_params}")

    trainable_params = vla_params + action_head_params + proprio_params
    # RDVLA_ENCODER_LR_SCALE (2026-11-03, mem-hvla), default off: the S1 vision encoders (self.vit, and
    # self.short_vit / self.hist_vit when present) get their own AdamW group at scale x the base LR, as
    # Zeng et al.'s encoder_lr_scale. Group 0 stays the base-LR group, so the logged "lr" is unchanged.
    _enc_scale = float(os.environ.get("RDVLA_ENCODER_LR_SCALE", "1") or 1)
    _s1m = getattr(getattr(action_head, "module", action_head), "model", None)
    if _enc_scale != 1.0 and _s1m is not None and hasattr(_s1m, "vit"):
        _enc = [q for name in ("vit", "short_vit", "hist_vit") if hasattr(_s1m, name)
                for q in getattr(_s1m, name).parameters() if q.requires_grad]
        _enc_ids = {id(q) for q in _enc}
        _rest = [q for q in trainable_params if id(q) not in _enc_ids]
        optimizer = AdamW([{"params": _rest, "lr": cfg.learning_rate, "lr_scale": 1.0},
                           {"params": _enc, "lr": cfg.learning_rate * _enc_scale, "lr_scale": _enc_scale}],
                          lr=cfg.learning_rate)
        print(f"[optim] vision-encoder LR scale {_enc_scale}: {sum(q.numel() for q in _enc):,} params at "
              f"{cfg.learning_rate * _enc_scale:.2e}, {sum(q.numel() for q in _rest):,} at {cfg.learning_rate:.2e}",
              flush=True)
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    optimizers = [optimizer]

    # RDVLA_EMA=1 (2026-11-03, mem-hvla), default off: an fp32 exponential moving average of every trainable
    # parameter with diffusion_policy's EMAModel schedule (decay = 1 - (1 + step)^-0.75, capped at 0.9999). Periodic
    # checkpoints keep the RAW weights (auto-resume reloads them) and carry the EMA in train_state; only the FINAL
    # checkpoint is written with the EMA weights, which is what the eval loads.
    ema = None
    if os.environ.get("RDVLA_EMA") == "1":
        ema = ts.ParamEMA(vla_params + action_head_params + proprio_params)
        print(f"[ema] tracking {sum(q.numel() for q in ema.params):,} params (power 0.75, max 0.9999)", flush=True)

    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # Restore optimiser moments, LR-schedule position and RNG. Weights were already restored
    # above: action head + proprio by init_module, the VLM's LoRA adapter + action_queries in
    # the use_lora branch (before 2026-10-06 the VLM was NOT restored -- see the fix there).
    start_step = 0
    if resume_blob is not None:
        start_step = ts.restore_into(resume_blob, optimizers, scheduler, device_id)
        if ema is not None:
            _e = (resume_blob.get("extra") or {}).get("ema")
            if _e is None:
                raise RuntimeError("RDVLA_EMA=1 but the train_state being resumed holds no EMA")
            ema.load_state_dict(_e)
            print(f"[resume] EMA restored at update {ema.num_updates}", flush=True)
        print(f"[resume] optimiser + scheduler + RNG restored; resuming at step {start_step} "
              f"of {cfg.max_steps}", flush=True)
    elif getattr(cfg, "resume_weights_only", False):
        start_step = int(cfg.resume_step)
        for _ in range(start_step):          # put the LR schedule where the step counter is
            scheduler.step()
        print(f"[resume] weights-only: fresh optimiser, schedule advanced to step {start_step} "
              f"(lr {optimizer.param_groups[0]['lr']:.2e}) of {cfg.max_steps}", flush=True)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer, image_transform=processor.image_processor.apply_transform,
        use_wrist_image=use_wrist_image, use_proprio=cfg.use_proprio,
    )

    dataset_fraction = getattr(cfg, 'dataset_fraction', 1.0)
    train_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug,
        dataset_fraction=dataset_fraction,
    )

    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir, cfg.dataset_name, batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10, image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    if distributed_state.is_main_process:
        print(f'Dataset size: {len(dataloader)} batches')

    if cfg.use_val_set:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0,
        )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        # Present only when the head returns one prediction per iterate (deep supervision), so their presence in the
        # logs is the proof that branch is active.
        "deep_supervision_n": deque(maxlen=cfg.grad_accumulation_steps),
        "iter_0_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "iter_final_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # File logging for runs without wandb (rectified-recurrence project): one JSON line per optimizer step, plus a
    # recurrence-health probe every RDVLA_HEALTH_EVERY steps.
    import json
    metrics_path = os.environ.get("RDVLA_METRICS_FILE", "")
    health_every = int(os.environ.get("RDVLA_HEALTH_EVERY", "0"))
    last_health = None
    train_t0 = time.time()
    # Held-out depth curve every RDVLA_DEPTH_EVAL_EVERY steps on RDVLA_DEPTH_EVAL_BATCHES fixed validation batches.
    depth_eval_every = int(os.environ.get("RDVLA_DEPTH_EVAL_EVERY", "0"))
    depth_eval_batches = []
    if depth_eval_every > 0 and distributed_state.is_main_process:
        depth_val_kwargs = dict(resize_resolution=tuple(vla.module.config.image_sizes), shuffle_buffer_size=1000,
                                image_aug=False)
        try:  # LIBERO's RLDS build has no validation split; fall back to held-back training batches there.
            depth_val_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, train=False,
                                            **depth_val_kwargs)
        except ValueError as e:
            print(f"[depth eval] no validation split ({e}); using training batches instead", flush=True)
            depth_val_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, train=True,
                                            **depth_val_kwargs)
        depth_val_loader = DataLoader(depth_val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
        for _, batch in zip(range(int(os.environ.get("RDVLA_DEPTH_EVAL_BATCHES", "6"))), depth_val_loader):
            depth_eval_batches.append(batch)
        del depth_val_loader, depth_val_dataset
        print(f"[depth eval] cached {len(depth_eval_batches)} validation batches of {cfg.batch_size}", flush=True)
    # Stop (with a final checkpoint) if the recurrent state's share at the adapter stays below RDVLA_COLLAPSE_SHARE
    # for RDVLA_COLLAPSE_PATIENCE consecutive health probes.
    collapse_share = float(os.environ.get("RDVLA_COLLAPSE_SHARE", "0"))
    collapse_patience = int(os.environ.get("RDVLA_COLLAPSE_PATIENCE", "3"))
    low_share_probes = 0
    stop_training = False

    with tqdm.tqdm(total=cfg.max_steps, leave=False, disable=not distributed_state.is_main_process) as progress:
        vla.train()
        for opt in optimizers:
            opt.zero_grad()

        batch_idx = 0
        # REMAINING work, not total: a continuation that re-ran cfg.max_steps would train the
        # whole schedule again on top of what the previous segment already did.
        max_batch_idx = max(0, cfg.max_steps - start_step) * cfg.grad_accumulation_steps
        if start_step > 0:
            progress.update(start_step)
        # RDVLA_TIMING=1: log data-wait vs compute seconds (each row covers the micro-batches since the last row); RDVLA_PREFETCH=N: prepare N batches ahead.
        timing = os.environ.get("RDVLA_TIMING") == "1"
        prefetch = int(os.environ.get("RDVLA_PREFETCH", "0"))
        data_s = compute_s = 0.0
        t_mark = time.perf_counter()

        while batch_idx < max_batch_idx:
            for batch in (_prefetch_batches(dataloader, prefetch) if prefetch > 0 else dataloader):
                if batch_idx >= max_batch_idx:
                    break
                if timing:
                    t_got = time.perf_counter()
                    data_s += t_got - t_mark

                do_health = health_every > 0 and batch_idx % (health_every * cfg.grad_accumulation_steps) == 0
                loss, metrics = run_forward_pass(
                    vla=vla, action_head=action_head, proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch, action_tokenizer=action_tokenizer, device_id=device_id,
                    use_proprio=cfg.use_proprio,
                    num_patches=NUM_PATCHES, cfg=cfg, health=do_health,
                )
                if "health" in metrics:
                    last_health = metrics.pop("health")

                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                log_step = start_step + gradient_step_idx

                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr * param_group.get("lr_scale", 1.0)

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    all_params = vla_params + action_head_params + proprio_params
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                    for opt in optimizers:
                        opt.step()
                    if ema is not None:
                        ema.update()
                    scheduler.step()
                    for opt in optimizers:
                        opt.zero_grad()
                    progress.update()

                    if distributed_state.is_main_process:
                        smoothened = {k: sum(v) / len(v) for k, v in recent_metrics.items() if v}
                        if timing:
                            smoothened.update(data_wait_s=round(data_s, 4), compute_s=round(compute_s, 4))
                            data_s = compute_s = 0.0
                        row = {"step": log_step, "time_s": round(time.time() - train_t0, 1),
                               "lr": optimizer.param_groups[0]["lr"], **smoothened,
                               "gpu_peak_gb": round(torch.cuda.max_memory_allocated(device_id) / 2**30, 2),
                               "gpu_reserved_gb": round(torch.cuda.max_memory_reserved(device_id) / 2**30, 2)}
                        if last_health is not None:
                            row.update(last_health)
                        if metrics_path:
                            with open(metrics_path, "a") as f:
                                f.write(json.dumps(row) + "\n")
                        if cfg.use_wandb and (log_step % cfg.wandb_log_freq == 0 or last_health is not None):
                            payload = {f"VLA Train/{k}": v for k, v in smoothened.items()}
                            payload["VLA Train/lr"] = row["lr"]
                            payload["VLA Train/hours"] = row["time_s"] / 3600
                            if last_health is not None:
                                payload.update({f"Health/{k}": v for k, v in last_health.items()})
                            wandb.log(payload, step=log_step)
                        if last_health is not None and collapse_share > 0:
                            low_share_probes = low_share_probes + 1 if last_health["state_share"] < collapse_share else 0
                            if low_share_probes >= collapse_patience:
                                print(f"[collapse] state share < {collapse_share} for {low_share_probes} probes at step "
                                      f"{log_step}; stopping with a final checkpoint", flush=True)
                                if cfg.use_wandb:
                                    wandb.log({"Health/collapse_stop": 1}, step=log_step)
                                stop_training = True
                        last_health = None

                # Once per optimizer step (first micro-batch), not once per micro-batch.
                at_step_start = batch_idx % cfg.grad_accumulation_steps == 0
                if at_step_start and depth_eval_batches and log_step % depth_eval_every == 0:
                    # Flow heads have no recurrent depth; their comparable axis is NFE.
                    _probe = depth_eval if _supports_recurrent_probe(
                        action_head.module if hasattr(action_head, "module") else action_head
                    ) else flow_eval
                    depth = _probe(
                        vla, action_head, proprio_projector if cfg.use_proprio else None, depth_eval_batches,
                        device_id, NUM_PATCHES,
                    )
                    print(f"[depth eval step {log_step}] " + ", ".join(f"{k}={v:.4f}" for k, v in depth.items()), flush=True)
                    if metrics_path:
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps({"step": log_step, "depth_eval": depth}) + "\n")
                    if cfg.use_wandb:
                        wandb.log({f"Depth/{k}": v for k, v in depth.items()}, step=log_step)

                if at_step_start and gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                    _ckpt_dir, _ckpt_tag = save_training_checkpoint(
                        cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla, processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        action_head=action_head, train_dataset=train_dataset,
                        distributed_state=distributed_state, new_state_dict=RAW_STATE_DICT,
                    )
                    if distributed_state.is_main_process:
                        ts.save(run_dir, step=log_step, checkpoint_dir=_ckpt_dir,
                                optimizers=optimizers, scheduler=scheduler,
                                extra={"checkpoint_tag": _ckpt_tag, "max_steps": cfg.max_steps,
                                       **({"ema": ema.state_dict()} if ema is not None else {})})
                    dist.barrier()

                # Myriad sends SIGTERM at h_rt minus the margin, then SIGKILL. Finishing the
                # micro-batch, saving, and exiting with the continue-code hands the chain a
                # clean restart point instead of being killed mid-write.
                if at_step_start and ts.deadline_passed(
                        margin_s=float(os.environ.get("RDVLA_DEADLINE_MARGIN_S", "420"))):
                    print(f"[myriad] deadline reached at step {log_step}; saving and exiting "
                          f"with the continue code (99)", flush=True)
                    _ckpt_dir, _ckpt_tag = save_training_checkpoint(
                        cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla, processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        action_head=action_head, train_dataset=train_dataset,
                        distributed_state=distributed_state, new_state_dict=RAW_STATE_DICT,
                    )
                    if distributed_state.is_main_process:
                        ts.save(run_dir, step=log_step, checkpoint_dir=_ckpt_dir,
                                optimizers=optimizers, scheduler=scheduler,
                                extra={"checkpoint_tag": _ckpt_tag, "max_steps": cfg.max_steps,
                                       **({"ema": ema.state_dict()} if ema is not None else {})})
                        ts.write_provenance(run_dir, cfg, step=log_step,
                                            note="segment end: deadline, exit 99")
                    dist.barrier()
                    if cfg.use_wandb and distributed_state.is_main_process:
                        wandb.finish()
                    # torchrun does NOT propagate a child's exit code: a non-zero child
                    # becomes ChildFailedError and torchrun exits 1, which Myriad reads as
                    # FAILED and cancels the rest of the chain -- the exact opposite of
                    # "continue". So leave a sentinel the launcher turns back into 99.
                    _sentinel = os.environ.get("RDVLA_CONTINUE_SENTINEL")
                    if _sentinel and distributed_state.is_main_process:
                        Path(_sentinel).parent.mkdir(parents=True, exist_ok=True)
                        Path(_sentinel).write_text(str(log_step))
                        print(f"[myriad] wrote continue sentinel {_sentinel}", flush=True)
                    sys.exit(99)

                if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                    run_validation(
                        vla=vla, action_head=action_head,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        val_dataloader=val_dataloader, action_tokenizer=action_tokenizer,
                        device_id=device_id, cfg=cfg, num_patches=NUM_PATCHES,
                        log_step=log_step, distributed_state=distributed_state,
                        val_time_limit=cfg.val_time_limit,
                    )

                batch_idx += 1
                if timing:  # compute = batch in hand -> end of this micro-batch (forward, backward, step, logging)
                    torch.cuda.synchronize()
                    t_mark = time.perf_counter()
                    compute_s += t_mark - t_got

                if log_step == cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break

                if stop_training:
                    max_batch_idx = batch_idx
                    break

            if batch_idx >= max_batch_idx:
                break

        final_step = start_step + gradient_step_idx
        if os.environ.get("RDVLA_SKIP_FINAL_SAVE") == "1":  # throughput benchmarks: no 2.7 GB checkpoint
            print(f"Training complete at step {final_step}. RDVLA_SKIP_FINAL_SAVE=1, not saving.")
            return
        if distributed_state.is_main_process:
            print(f"Training complete at step {final_step}. Saving final checkpoint...")
        if ema is not None:      # the final checkpoint carries the EMA weights -- the ones the eval should load
            ema.copy_to_params()
            print(f"[ema] final checkpoint written with EMA weights (update {ema.num_updates}, "
                  f"decay {ema.decay():.5f})", flush=True)
        _ckpt_dir, _ckpt_tag = save_training_checkpoint(
            cfg=cfg, run_dir=run_dir, log_step=final_step, vla=vla, processor=processor,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            action_head=action_head, train_dataset=train_dataset,
            distributed_state=distributed_state, new_state_dict=RAW_STATE_DICT,
        )
        # The final save needs a train_state too. Without this the last state on disk is from
        # the previous PERIODIC save, so an ungraceful kill (node failure) after training ends
        # but before the job exits would resume from up to save_freq steps earlier, and a
        # resume would silently target an older checkpoint than the final one.
        # Note log_step never actually reaches max_steps -- the loop tops out at
        # gradient_step_idx = max_steps - 1 -- so the periodic branch never fires on the last
        # step and this is the only state written for the final interval.
        if distributed_state.is_main_process:
            ts.save(run_dir, step=final_step, checkpoint_dir=_ckpt_dir,
                    optimizers=optimizers, scheduler=scheduler,
                    extra={"checkpoint_tag": _ckpt_tag, "max_steps": cfg.max_steps,
                           "final": True})
            ts.write_provenance(run_dir, cfg, step=final_step, note="training complete")


def main(argv=None):
    """python -m recurrent_vla.train --config CONFIG.yaml [--key=value ...]"""
    import argparse

    from recurrent_vla.config import TrainConfig, get_legacy_config, load_config, save_config

    parser = argparse.ArgumentParser(description="Train an HVLA policy (evaluate with recurrent_vla.eval.run_libero_eval)")
    parser.add_argument("--config", type=str, required=True, help="YAML config, e.g. recurrent_vla/configs/libero/hvla_recurrent_long.yaml")
    args, overrides = parser.parse_known_args(argv)
    cfg = load_config(TrainConfig, args.config, overrides)
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    finetune(get_legacy_config(cfg, "train"))


if __name__ == "__main__":
    main()
