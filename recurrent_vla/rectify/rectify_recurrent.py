"""Rectify RD-VLA's recurrent core so 1-2 iterations reach the teacher's converged latent.

Frozen: VLM (merged LoRA), proprio projector, prelude, coda, action queries.
Trained: adapter, adapter_norm, gamma_adapt, recurrent layers, recurrent_norm (float32 master weights, bf16 autocast).

Loss for a batch, teacher and student starting from the SAME noise S_0:
  endpoint : for k in --student_steps (default 1,2):
               ||S_k^stu - S_T^tea||^2 / ||S_T^tea||^2          (latent, T = --teacher_iters)
             + L1(coda(S_k^stu), coda(S_T^tea))                  (teacher actions)
             + w_gt * L1(coda(S_k^stu), a_gt)                    (ground truth)
  anchor   : ||step^stu(S_T^tea) - S_T^tea||^2 / ||S_T^tea||^2   (the converged state stays a fixed point)
  deep     : w_deep * L1(coda(S_k^stu), a_gt) for k in --deep_steps (extra iterations stay useful)

Defaults are the paper's student recipe: latent + teacher-action only (w_gt = w_anchor = w_deep = 0;
the two-term loss matched or beat the five-term one, and the anchor conflicted with the endpoint),
10,000 steps on a fixed 10,000-example pool (1250 batches x bs 8, each example seen 8x).

The output dir mirrors the original checkpoint with the student head under the original filename, so
experiments/robot/libero/run_libero_eval.py loads it unchanged.

  python -m recurrent_vla.rectify.rectify_recurrent --ckpt outputs/12_24-24_24_Spatial_40k \
      --dataset_name libero_spatial_no_noops --out outputs/rect_k12 --steps 3000
  (--eval_only prints the teacher's offline iteration curve.) The dataset name must be on the command line:
  prismatic picks the LIBERO constants from argv.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


import torch
import torch.nn.functional as F

from recurrent_vla.rectify import frozen_backbone as fb

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--data_root_dir", default="datasets/libero")
p.add_argument("--dataset_name", required=True)
p.add_argument("--out", default=None)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--teacher_iters", type=int, default=32)
p.add_argument("--student_steps", default="1,2")
p.add_argument("--deep_steps", default="4,8")
p.add_argument("--w_latent", type=float, default=1.0)
p.add_argument("--w_teacher_action", type=float, default=1.0)
p.add_argument("--w_gt", type=float, default=0.0)
p.add_argument("--w_anchor", type=float, default=0.0)
p.add_argument("--w_deep", type=float, default=0.0)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--eval_batches", type=int, default=20)
p.add_argument("--eval_only", action="store_true")
p.add_argument("--shuffle_buffer", type=int, default=20000)
p.add_argument("--seed", type=int, default=0)
# Ablation: where the student's starting noise S_0 comes from.
#   fresh  - redraw S_0 every time a sample is seen (current behaviour)
#   pinned - derive S_0 from a fingerprint of the sample itself, so the SAME example
#            always gets the SAME noise, however often it recurs. This is the reflow
#            analogue: one noise permanently bound to one target.
# Only meaningful with repetition (multiple epochs over the data); with a single pass
# each example is seen once and the two modes are indistinguishable.
p.add_argument("--noise_mode", choices=["fresh", "pinned"], default="fresh")
# Independent S_0 draws used by offline_curve to measure noise robustness (0/1 = off).
# This is the signature the noise_mode ablation predicts, so it is measured, not inferred.
p.add_argument("--spread_draws", type=int, default=4)
# Cache this many batches once and cycle them, instead of streaming fresh data.
# The noise_mode ablation needs REPETITION to mean anything: at the standard 3000 steps x bs 8
# = 24,000 presentations against LIBERO-Long's 101,469 transitions, each example is seen 0.24
# times, so fresh and pinned are indistinguishable. A small pool gives repetition at the SAME
# proven budget, and it stresses the hypothesis hardest -- pinning should hurt most when noise
# diversity is scarce. 375 batches = 3000 examples, each seen 8x over 3000 steps (~7 GB RAM).
# Caching post-transform also freezes the pixels, so the ONLY difference between arms is S_0.
# Default 1250 batches = the paper's 10,000-example pool (~23 GB RAM). 0 streams fresh data instead.
p.add_argument("--pool_batches", type=int, default=1250)
args = p.parse_args()

torch.manual_seed(args.seed)
dev = torch.device("cuda")
student_steps = [int(s) for s in args.student_steps.split(",") if s]
deep_steps = [int(s) for s in args.deep_steps.split(",") if s] if args.w_deep > 0 else []  # unweighted: don't iterate past k=2
ckpt = Path(args.ckpt)

vla, processor, proprio_projector, num_patches = fb.load_frozen_backbone(ckpt, dev)
teacher, head_fname = fb.load_recurrent_head(ckpt, vla.llm_dim, dev)
student = copy.deepcopy(teacher).float()
TRAIN_KEYS = ("adapter", "adapter_norm", "gamma_adapt", "recurrent.", "recurrent_norm")
trainable = []
for name, prm in student.model.named_parameters():
    prm.requires_grad_(any(name.startswith(k) for k in TRAIN_KEYS))
    if prm.requires_grad:
        trainable.append(prm)
print(f"trainable params: {sum(q.numel() for q in trainable) / 1e6:.1f}M of {sum(q.numel() for q in student.parameters()) / 1e6:.1f}M")
loader = fb.make_loader(vla, processor, args.data_root_dir, args.dataset_name, args.bs, args.shuffle_buffer)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


IS_HVLA = fb.is_hvla_head(teacher)


def make_ops(m, ctx_a, ctx_b, prop):
    """Per-(model, batch) rollout hooks: (build_prelude, iterate, decode).

    Two problems solved in one place.

    1. Interface. The stock and S1-recurrent heads take (h_a, h_t, p); the full-HVLA heads take
       (pixel_values, s2_latent, p). Inlining either one silently distils against a target the
       model never produces -- the expensive kind of wrong.

    2. Cost. For HVLA, `_run_one_iteration` / `_get_output` deliberately REBUILD the context on
       every call (they are written that way so no stale context can leak between batches).
       Rectification does teacher_iters (32) + kmax (8) + 1 anchor calls per step, so that is
       ~41 DINOv2 ViT-S + observation-encoder forwards per step to recompute an identical
       tensor. Here the call sequence is under our control and the encoder is frozen, so the
       context is built ONCE per batch and the lower-level hooks are used directly.

    Built under no_grad: the ViT, projections and obs_encoder are all frozen (TRAIN_KEYS covers
    only the recurrent core), and gradients still reach the adapter/recurrent weights through
    the cross-attention that consumes these tensors.
    """
    if IS_HVLA:
        with torch.no_grad():
            ctx_v, ctx_l = m.build_context(ctx_a, ctx_b, prop)
        return (lambda: m.build_prelude_ctx(ctx_v, ctx_l, prop, ctx_v.dtype),
                lambda s, pre: m.iterate(s, pre, ctx_v, ctx_l, prop),
                lambda s: m.decode(s, ctx_v, ctx_l, prop))
    return (lambda: m.build_prelude(ctx_a, ctx_b, prop),
            lambda s, pre: m._run_one_iteration(s, pre, ctx_a, ctx_b, prop),
            lambda s: m._get_output(s, ctx_a, ctx_b, prop))


def sample_fingerprints(gt, prop):
    """Stable per-sample ids from the sample's own content, not its position in the stream.

    RLDS shuffles and interleaves, so a sample has no stable index across epochs. Hashing
    (actions, proprio) gives an order-independent identity with no caching, which is what
    lets the pinned mode run over the FULL dataset rather than a cached subset.

    Note this deliberately ignores pixels: with image_aug on, the same underlying example
    yields different pixels each epoch but the same actions/proprio, so it keeps the same
    noise -- which is exactly the intended meaning of "this example's noise".
    """
    a = gt.detach().float().cpu().numpy()
    p = prop.detach().float().cpu().numpy()
    return [int.from_bytes(hashlib.blake2b(a[i].tobytes() + p[i].tobytes(), digest_size=8).digest(),
                           "little") % (2 ** 31) for i in range(a.shape[0])]


def make_s0(T, B, dtype, fingerprints=None):
    """S_0 for the student/teacher rollout. Identical distribution in both modes.

    The pinned branch reproduces VLARecurrent.init_state exactly -- same std
    (gamma_init * state_init_std, read from the module rather than assumed), same truncated
    normal, same shape -- and differs ONLY in the seed. Anything else would make the two arms
    incomparable for reasons unrelated to the coupling.
    """
    if fingerprints is None:
        return T.init_state(B, dev, dtype)
    base_std = getattr(T.cfg, "state_init_std", 0.0) or T.cfg.init_std
    std = (T.gamma_init * base_std).item()
    out = torch.empty(B, T.cfg.action_chunk_len, T.cfg.hidden_dim, device=dev, dtype=dtype)
    rng = torch.get_rng_state()
    try:
        for i, fp in enumerate(fingerprints):
            torch.manual_seed(fp)
            t = torch.empty(T.cfg.action_chunk_len, T.cfg.hidden_dim, dtype=torch.float32)
            torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-3 * std, b=3 * std)
            out[i] = t.to(device=dev, dtype=dtype)
    finally:
        torch.set_rng_state(rng)
    return out


def nmse(a, b):
    return ((a.float() - b.float()).pow(2).flatten(1).sum(1) / b.float().pow(2).flatten(1).sum(1).clamp_min(1e-6)).mean()


def teacher_rollout(ctx_a, ctx_b, prop, B, s0=None):
    T = teacher.model
    t_pre, t_iter, t_out = make_ops(T, ctx_a, ctx_b, prop)
    pre = t_pre()
    # Teacher and student must start from the SAME S_0 -- that shared start is what makes the
    # latent target meaningful. Caller supplies it in pinned mode; otherwise draw fresh.
    s0 = T.init_state(B, dev, torch.bfloat16) if s0 is None else s0
    st = s0
    for _ in range(args.teacher_iters):
        st = t_iter(st, pre)
    return pre, s0, st, t_out(st)


CURVE_K = (1, 2, 4, 8, 12, 32)


@torch.no_grad()
def offline_curve(head, n_batches, spread_batches=2):
    """Mean L1 to GT and to the teacher's T-step actions after k iterations of `head`.

    Also reports `noise_std`: how much the head's k-step output varies across INDEPENDENT
    S_0 draws on the same batch. That is the mechanism the noise_mode ablation predicts --
    a pinned-noise student should be markedly less noise-robust at small k, because it only
    ever trained against one S_0 per example while inference draws S_0 at random. Measured
    on the first `spread_batches` batches to keep the probe cheap.

    Always uses FRESH draws regardless of --noise_mode: this measures inference behaviour,
    and keeping it identical across arms is what makes the comparison fair.
    """
    tot = {k: [0.0, 0.0] for k in CURVE_K}
    spread = {k: [] for k in CURVE_K}
    it = iter(loader)
    for bi in range(n_batches):
        ctx_a, ctx_b, prop, gt = fb.extract_context(vla, proprio_projector, num_patches, next(it), dev, teacher)
        with autocast():
            pre, s0, _, a_t = teacher_rollout(ctx_a, ctx_b, prop, gt.size(0))
            _, h_iter, h_out = make_ops(head.model, ctx_a, ctx_b, prop)
            s = s0
            for k in range(1, max(CURVE_K) + 1):
                s = h_iter(s, pre)
                if k in tot:
                    a = h_out(s)
                    tot[k][0] += F.l1_loss(a.float(), gt.float()).item() / n_batches
                    tot[k][1] += F.l1_loss(a.float(), a_t.float()).item() / n_batches
            if args.spread_draws > 1 and bi < spread_batches:
                outs = {k: [] for k in CURVE_K}
                for _ in range(args.spread_draws):
                    sd = head.model.init_state(gt.size(0), dev, torch.bfloat16)
                    for k in range(1, max(CURVE_K) + 1):
                        sd = h_iter(sd, pre)
                        if k in outs:
                            outs[k].append(h_out(sd).float())
                for k in CURVE_K:
                    spread[k].append(torch.stack(outs[k]).std(0).mean().item())
    res = {k: {"l1_gt": v[0], "l1_teacher": v[1]} for k, v in tot.items()}
    for k in CURVE_K:
        if spread[k]:
            res[k]["noise_std"] = sum(spread[k]) / len(spread[k])
    return res


if args.eval_only:
    print(json.dumps({"teacher_offline_curve": offline_curve(teacher, args.eval_batches)}, indent=1))
    sys.exit(0)

out = Path(args.out)
fb.mirror_checkpoint(ckpt, out)
json.dump(vars(args), open(out / "rectify_args.json", "w"), indent=1)
opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
log, t0 = [], time.time()
data_iter = iter(loader)
# Fixed pool for the noise_mode ablation: cached once, then cycled. Repetition is what makes
# fresh-vs-pinned mean anything, and caching post-transform freezes the pixels too, so S_0 is
# the only thing that differs between arms.
pool = None
if args.pool_batches > 0:
    _it = iter(loader)
    pool = [next(_it) for _ in range(args.pool_batches)]
    del _it
    print(f"pool: {len(pool)} batches = {len(pool) * args.bs} examples; "
          f"{args.steps} steps -> each example seen {args.steps / len(pool):.1f}x "
          f"(noise_mode={args.noise_mode})", flush=True)
kmax = max(student_steps + deep_steps)
for step in range(args.steps + 1):
    if step % args.eval_every == 0 or step == args.steps:
        student.eval()
        curve = offline_curve(student, args.eval_batches)
        log.append({"step": step, "curve": curve, "t": time.time() - t0})
        print(f"[step {step}] offline L1-to-GT by k: " + "  ".join(f"k{k}={v['l1_gt']:.4f}" for k, v in curve.items()), flush=True)
        json.dump(log, open(out / "rectify_log.json", "w"), indent=1)
        if step == args.steps:
            break
        student.train()
    for g in opt.param_groups:
        g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    if pool is not None:
        batch = pool[step % len(pool)]
    else:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
    ctx_a, ctx_b, prop, gt = fb.extract_context(vla, proprio_projector, num_patches, batch, dev, teacher)
    S = student.model
    fps = sample_fingerprints(gt, prop) if args.noise_mode == "pinned" else None
    with autocast():
        with torch.no_grad():
            s0_in = make_s0(teacher.model, gt.size(0), torch.bfloat16, fps)
            pre, s0, st, a_t = teacher_rollout(ctx_a, ctx_b, prop, gt.size(0), s0=s0_in)
        _, s_iter, s_out = make_ops(S, ctx_a, ctx_b, prop)
        terms = {"latent": 0.0, "teacher_action": 0.0, "gt": 0.0, "deep": 0.0}
        s = s0
        for k in range(1, kmax + 1):
            s = s_iter(s, pre)
            if k in student_steps:
                a = s_out(s)
                terms["latent"] = terms["latent"] + nmse(s, st) / len(student_steps)
                terms["teacher_action"] = terms["teacher_action"] + F.l1_loss(a.float(), a_t.float()) / len(student_steps)
                terms["gt"] = terms["gt"] + F.l1_loss(a.float(), gt.float()) / len(student_steps)
            if k in deep_steps:
                terms["deep"] = terms["deep"] + F.l1_loss(s_out(s).float(), gt.float()) / len(deep_steps)
        anchor = nmse(s_iter(st, pre), st) if args.w_anchor > 0 else torch.zeros((), device=dev)
    loss = (args.w_latent * terms["latent"] + args.w_teacher_action * terms["teacher_action"] + args.w_gt * terms["gt"]
            + args.w_deep * terms["deep"] + args.w_anchor * anchor)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    if step % 50 == 0:
        print(json.dumps({"step": step, "loss": round(loss.item(), 5), "anchor": round(anchor.item(), 5),
                          **{k: round(float(v), 5) for k, v in terms.items()}, "t": round(time.time() - t0, 1)}), flush=True)

torch.save({k: v.to(torch.bfloat16) for k, v in student.state_dict().items()}, out / head_fname)
print(f"saved student head to {out / head_fname}")
