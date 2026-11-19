"""Ablation 3 (student width): train a recurrent student of width d FROM SCRATCH against a frozen teacher.

The student owns everything up to and including its recurrent state, all at width d = --loop_dim and randomly
initialised: action queries, prelude, adapter (+ norm, gamma), recurrent core (+ norm), and a learned projection
P: d -> 896 (`state_proj`). P maps the student's state S_k into the teacher's state space, where the TEACHER'S coda
(frozen, copied in) decodes the action. P is part of the model at evaluation. Everything the student reads is the
teacher's and frozen: the VLM features (RD-VLA) or the S1 observation context -- ViT-S + observation encoder -- (HVLA),
read through 896 -> d key/value projections, plus the proprio feature. So, for both architectures, only the
student's own prelude + loop scale with d, and capacity is the only thing the sweep varies.

Loss (the paper's two-term student recipe, carried to width d through P), for k in --student_steps (default 1,2):
    ||P S_k^stu - S_T^tea||^2 / ||S_T^tea||^2           (latent, T = --teacher_iters = 32)
  + L1(coda(P S_k^stu), coda(S_T^tea))                   (teacher actions)
The student starts from its OWN S_0 (width d); the teacher's S_T does not depend on its S_0 (fan-in ~700x), so
there is no shared-noise coupling to preserve -- noise is drawn fresh every step.

Budget as the spec: 10,000 steps on a fixed 10,000-example pool (1250 batches x bs 8), lr 1e-4 cosine, AdamW.
Fast path as rectify_fast.py: the pool's frozen VLM features are computed once and cached (see below).
Widths (multiples of 16 so the head dim is even for RoPE with 8 heads): 224 / 400 / 608 / 768 / 896 = 9.8 / 24.6 /
50.2 / 75.8 / 100 % of the student-owned parameters at d = 896 (55.4M incl. P).

The output dir mirrors the teacher checkpoint (VLM, LoRA, proprio projector symlinked) with the student head under
the teacher's head filename and an action_head_config carrying loop_dim, so recurrent_vla.eval.run_libero_eval
load it unchanged.

  python -m recurrent_vla.rectify.train_scratch_student --ckpt <80k teacher> --loop_dim 400 \
      --dataset_name libero_10_no_noops --out <dir> --seed 0
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


import torch
import torch.nn.functional as F

from recurrent_vla.rectify import frozen_backbone as fb
from recurrent_vla.models.recurrent_core import RecurrentConfigInternal

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="teacher checkpoint (80k)")
p.add_argument("--loop_dim", type=int, required=True, help="student width d (multiple of 16)")
p.add_argument("--data_root_dir", default="datasets/libero")
p.add_argument("--dataset_name", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--teacher_iters", type=int, default=32)
p.add_argument("--student_steps", default="1,2")
p.add_argument("--w_latent", type=float, default=1.0)
p.add_argument("--w_teacher_action", type=float, default=1.0)
p.add_argument("--pool_batches", type=int, default=1250)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--eval_batches", type=int, default=20)
p.add_argument("--shuffle_buffer", type=int, default=20000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
assert args.loop_dim % 16 == 0, "loop_dim must be a multiple of 16 (RoPE needs an even head dim with 8 heads)"

torch.manual_seed(args.seed)
dev = torch.device("cuda")
student_steps = [int(s) for s in args.student_steps.split(",") if s]
ckpt = Path(args.ckpt)
out = Path(args.out)

vla, processor, proprio_projector, num_patches = fb.load_frozen_backbone(ckpt, dev)
teacher, head_fname = fb.load_recurrent_head(ckpt, vla.llm_dim, dev)
teacher.eval().requires_grad_(False)
IS_HVLA = fb.is_hvla_head(teacher)

# ---- student: the teacher's head class at loop_dim = d; own parts random, the rest copied from the teacher and frozen
OWN = ("action_queries", "prelude.", "adapter", "gamma_adapt", "recurrent.", "recurrent_norm", "state_proj")
cfg_path = sorted(ckpt.glob("action_head_config--*.json"))[0]
saved = json.load(open(cfg_path))
head_type = saved.pop("_type")
for k in ("prelude_vlm_layers", "recurrent_vlm_layers", "coda_vlm_layers"):
    saved[k] = tuple(saved[k])
saved["loop_dim"] = args.loop_dim
student = type(teacher)(hidden_dim=vla.llm_dim, cfg=RecurrentConfigInternal(**saved)).to(dev).float()
t_sd = teacher.state_dict()
s_sd = student.state_dict()
copied = {k: v for k, v in t_sd.items() if not k.startswith(tuple("model." + o for o in OWN))}
missing = [k for k in copied if k not in s_sd or s_sd[k].shape != copied[k].shape]
assert not missing, f"teacher parts do not fit the student: {missing[:5]}"
student.load_state_dict(copied, strict=False)
trainable, n_frozen = [], 0
for name, prm in student.model.named_parameters():
    own = name.startswith(OWN)
    prm.requires_grad_(own)
    if own:
        trainable.append(prm)
    else:
        n_frozen += prm.numel()
own_n = sum(q.numel() for q in trainable)
print(f"[student] {head_type} loop_dim={args.loop_dim}: trainable (from scratch) {own_n / 1e6:.2f}M; "
      f"frozen copy of the teacher {n_frozen / 1e6:.2f}M (coda{' + S1 context encoder' if IS_HVLA else ''})", flush=True)
assert hasattr(student.model, "state_proj"), "student must carry P"

loader = fb.make_loader(vla, processor, args.data_root_dir, args.dataset_name, args.bs, args.shuffle_buffer)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def contexts(ctx_a, ctx_b, prop):
    """The (frozen) context both models read. HVLA: build the S1 context ONCE with the teacher's encoder (the student
    holds an identical frozen copy, so its own encoder would produce the same tensors)."""
    if IS_HVLA:
        with torch.no_grad():
            ctx_v, ctx_l = teacher.model.build_context(ctx_a, ctx_b, prop)
        return ctx_v, ctx_l
    return ctx_a, ctx_b


def ops(m, ctx, prop):
    """(prelude, iterate, decode) for model m over a fixed context."""
    c1, c2 = ctx
    if IS_HVLA:
        return (lambda: m.build_prelude_ctx(c1, c2, prop, c1.dtype),
                lambda s, pre: m.iterate(s, pre, c1, c2, prop),
                lambda s: m.decode(s, c1, c2, prop))
    return (lambda: m.build_prelude(c1, c2, prop),
            lambda s, pre: m._run_one_iteration(s, pre, c1, c2, prop),
            lambda s: m._get_output(s, c1, c2, prop))


@torch.no_grad()
def teacher_target(ctx, prop, B):
    pre_f, it, dec = ops(teacher.model, ctx, prop)
    pre = pre_f()
    s = teacher.model.init_state(B, dev, torch.bfloat16)
    for _ in range(args.teacher_iters):
        s = it(s, pre)
    return s, dec(s)


def nmse(a, b):
    return ((a.float() - b.float()).pow(2).flatten(1).sum(1) / b.float().pow(2).flatten(1).sum(1).clamp_min(1e-6)).mean()


CURVE_K = (1, 2, 4, 8)


@torch.no_grad()
def offline_curve(n_batches):
    """Student's offline L1 (to GT and to the teacher's 32-step action) and latent NMSE after k of its own iterations."""
    tot = {k: [0.0, 0.0, 0.0] for k in CURVE_K}
    it_ = iter(loader)
    for _ in range(n_batches):
        ctx_a, ctx_b, prop, gt = fb.extract_context(vla, proprio_projector, num_patches, next(it_), dev, teacher)
        with autocast():
            ctx = contexts(ctx_a, ctx_b, prop)
            st, a_t = teacher_target(ctx, prop, gt.size(0))
            pre_f, s_it, s_dec = ops(student.model, ctx, prop)
            pre = pre_f()
            s = student.model.init_state(gt.size(0), dev, torch.bfloat16)
            for k in range(1, max(CURVE_K) + 1):
                s = s_it(s, pre)
                if k in tot:
                    a = s_dec(s)
                    tot[k][0] += F.l1_loss(a.float(), gt.float()).item() / n_batches
                    tot[k][1] += F.l1_loss(a.float(), a_t.float()).item() / n_batches
                    tot[k][2] += nmse(student.model.state_proj(s), st).item() / n_batches
    return {k: {"l1_gt": v[0], "l1_teacher": v[1], "latent_nmse": v[2]} for k, v in tot.items()}


fb.mirror_checkpoint(ckpt, out)
# the mirrored config is the teacher's; the student's must carry loop_dim or every loader rebuilds a 896 head
cfg_out = out / cfg_path.name
if cfg_out.exists() or cfg_out.is_symlink():
    cfg_out.unlink()
json.dump({**{k: (list(v) if isinstance(v, tuple) else v) for k, v in saved.items()}, "_type": head_type},
          open(cfg_out, "w"), indent=2)
json.dump({**vars(args), "trainable_params": own_n, "head_type": head_type}, open(out / "scratch_args.json", "w"), indent=1)

opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
log, t0 = [], time.time()
pool = None
if args.pool_batches > 0:
    _it = iter(loader)
    pool = [next(_it) for _ in range(args.pool_batches)]
    del _it
    print(f"pool: {len(pool)} batches = {len(pool) * args.bs} examples; {args.steps} steps -> each example seen "
          f"{args.steps / len(pool):.1f}x", flush=True)
if pool is None:
    raise SystemExit("train_scratch_student.py caches the pool's VLM features: needs --pool_batches > 0")

# ---- FAST PATH (as rectify_fast.py): the VLM is frozen and eval, so a pool batch's features are the same tensor on
# every visit -- hoist that forward out of the step loop. RD-VLA: the scratch student runs its OWN prelude, which reads
# VLM layer 11, and the recurrent core + coda read layer 23, so cache exactly those layers (16.5 MB/batch, kept in
# pinned CPU RAM and moved per step) and point the layer indices at the compact stack during training. HVLA: cache the
# pooled S2 latent; the S1 context is rebuilt per step from the pixels by the teacher's frozen encoder.
T_m, S_m = teacher.model, student.model
_REAL = {m: (list(m.prelude_vlm_layers), list(m.recurrent_vlm_layers), list(m.coda_vlm_layers)) for m in (T_m, S_m)}
LAYERS = sorted({*_REAL[T_m][0], *_REAL[T_m][1], *_REAL[T_m][2]})
_POS = {l: i for i, l in enumerate(LAYERS)}


def remap(compact):
    for m in (T_m, S_m):
        pre_l, rec_l, cod_l = _REAL[m]
        f = (lambda ls: [_POS[l] for l in ls]) if compact else list
        m.prelude_vlm_layers, m.recurrent_vlm_layers, m.coda_vlm_layers = f(pre_l), f(rec_l), f(cod_l)


fcache = []
for bi, b in enumerate(pool):
    _ca, _cb, _prop, _gt = fb.extract_context(vla, proprio_projector, num_patches, b, dev, teacher)
    if IS_HVLA:
        fcache.append((b["pixel_values"], _cb.cpu().pin_memory(), _prop.cpu().pin_memory(), _gt.cpu().pin_memory()))
    else:
        fcache.append((_ca[:, LAYERS].contiguous().cpu().pin_memory(), _cb[:, LAYERS].contiguous().cpu().pin_memory(),
                       _prop.cpu().pin_memory(), _gt.cpu().pin_memory()))
    del _ca, _cb
    if (bi + 1) % 250 == 0:
        print(f"[fast] cached {bi + 1}/{len(pool)}", flush=True)
_nb = sum(t.numel() * t.element_size() for e in fcache for t in e[1:] if torch.is_tensor(t))
print(f"[fast] {'HVLA' if IS_HVLA else 'stock'}: cached {len(fcache)} batches, {_nb / 1e9:.2f} GB in CPU RAM "
      f"(layers {LAYERS if not IS_HVLA else 'pooled S2'}); the per-step VLM forward is gone", flush=True)
del pool
if not IS_HVLA:
    remap(True)

kmax = max(student_steps)
for step in range(args.steps + 1):
    if step % args.eval_every == 0 or step == args.steps:
        student.eval()
        if not IS_HVLA:
            remap(False)                 # offline_curve feeds the full per-layer stack
        curve = offline_curve(args.eval_batches)
        if not IS_HVLA:
            remap(True)
        log.append({"step": step, "curve": curve, "t": time.time() - t0})
        print(f"[step {step}] offline L1-to-GT by k: " + "  ".join(f"k{k}={v['l1_gt']:.4f}" for k, v in curve.items())
              + f"   latent nmse k1={curve[1]['latent_nmse']:.4f}", flush=True)
        json.dump(log, open(out / "scratch_log.json", "w"), indent=1)
        if step == args.steps:
            break
        student.train()
    for g in opt.param_groups:
        g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    e = fcache[step % len(fcache)]
    if IS_HVLA:
        ctx_a = e[0].to(torch.bfloat16).to(dev)          # extract_context's ctx_a is exactly this cast-and-move
        ctx_b, prop, gt = (t.to(dev, non_blocking=True) for t in e[1:])
    else:
        ctx_a, ctx_b, prop, gt = (t.to(dev, non_blocking=True) for t in e)
    with autocast():
        ctx = contexts(ctx_a, ctx_b, prop)
        st, a_t = teacher_target(ctx, prop, gt.size(0))
        pre_f, s_it, s_dec = ops(student.model, ctx, prop)
        pre = pre_f()
        s = student.model.init_state(gt.size(0), dev, torch.bfloat16)
        terms = {"latent": 0.0, "teacher_action": 0.0}
        for k in range(1, kmax + 1):
            s = s_it(s, pre)
            if k in student_steps:
                terms["latent"] = terms["latent"] + nmse(student.model.state_proj(s), st) / len(student_steps)
                terms["teacher_action"] = terms["teacher_action"] + F.l1_loss(s_dec(s).float(), a_t.float()) / len(student_steps)
    loss = args.w_latent * terms["latent"] + args.w_teacher_action * terms["teacher_action"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    if step % 50 == 0:
        print(json.dumps({"step": step, "loss": round(loss.item(), 5), **{k: round(float(v), 5) for k, v in terms.items()},
                          "t": round(time.time() - t0, 1)}), flush=True)

if not IS_HVLA:
    remap(False)                         # never save or reload with the compact indices
torch.save({k: v.to(torch.bfloat16) for k, v in student.state_dict().items()}, out / head_fname)
# round-trip: the saved config + weights must rebuild this exact head, strictly, as the eval loaders do
re_head, _ = fb.load_recurrent_head(out, vla.llm_dim, "cpu")
print(f"saved student head to {out / head_fname} (strict reload OK, loop_dim={re_head.model.loop_dim})", flush=True)
