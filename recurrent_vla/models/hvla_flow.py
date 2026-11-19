"""HVLA with a flow-matching decoder (config type `hvla_flow`): the non-recurrent baseline.

The S1 context is exactly the recurrent head's (hvla_context.S1ContextMixin, including the history
and mem-hvla modes); only the decoder differs. The recurrent core is replaced by
`flow_num_layers` (default 4) UNTIED decoder layers conditioned on a flow timestep. With 4 layers,
one denoising step (one NFE) costs about one recurrent iteration, so NFE and k are comparable.

Training calls `training_loss` (rectified-flow MSE; t = 0 is noise, t = 1 the action chunk), never
`predict_action`, which is a no-grad Euler integration. finetune.py's `flow_eval` sweeps NFE the
way `depth_eval` sweeps k.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from recurrent_vla.models.recurrent_core import RMSNorm, RecurrentConfigInternal, RecurrentLayer
from recurrent_vla.models.hvla_context import S1ContextMixin, pool_prefix


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class S1HVLAFlow(S1ContextMixin, nn.Module):
    """Flow-matching velocity field over the S1 context."""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        dim = cfg.hidden_dim
        self.action_chunk_len = cfg.action_chunk_len
        self.action_dim = cfg.action_dim
        self.num_inference_steps = int(getattr(cfg, "flow_num_inference_steps", 10))
        self._init_s1_context(cfg)

        n_layers = int(getattr(cfg, "flow_num_layers", 4))
        self.action_in = nn.Linear(cfg.action_dim, dim)
        self.queries = nn.Parameter(torch.randn(cfg.action_chunk_len, dim) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(
            [RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base) for _ in range(n_layers)])
        self.out_norm = RMSNorm(dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(dim, cfg.action_dim)

        # Scaled trunc-normal on the decoder ONLY (a walk over self.modules() would overwrite the
        # pretrained ViT and re-initialise the observation encoder the recurrent head keeps at its
        # defaults), and a zero output head so the initial velocity field is exactly zero.
        std = (2.0 / (5.0 * dim)) ** 0.5
        for mod in (self.action_in, self.time_mlp, self.layers, self.out_proj):
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def velocity_ctx(self, x_t, t, ctx_v, ctx_l, p):
        temb = self.time_mlp(timestep_embedding(t).to(x_t.dtype))
        h = self.action_in(x_t) + self.queries.unsqueeze(0).to(x_t.dtype) + temb[:, None, :]
        for layer in self.layers:
            h = layer(h, ctx_l, ctx_v, p)
        return self.out_proj(self.out_norm(h))

    def loss(self, actions, pixel_values, s2_latent, p):
        ctx_v, ctx_l = self.build_context(pixel_values, s2_latent, p)
        x1 = actions
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.size(0), device=x1.device)
        tt = t.to(x1.dtype)[:, None, None]
        x_t = (1 - tt) * x0 + tt * x1
        v = self.velocity_ctx(x_t, t, ctx_v, ctx_l, p)
        return (v.float() - (x1 - x0).float()).pow(2).mean()

    @torch.no_grad()
    def sample(self, pixel_values, s2_latent, p, nfe: int):
        ctx_v, ctx_l = self.build_context(pixel_values, s2_latent, p)
        B = ctx_v.size(0)
        x = torch.randn(B, self.action_chunk_len, self.action_dim, device=ctx_v.device, dtype=ctx_v.dtype)
        for i in range(nfe):
            t = torch.full((B,), i / nfe, device=ctx_v.device)
            x = x + self.velocity_ctx(x, t, ctx_v, ctx_l, p) / nfe
        return x

    def forward(self, pixel_values, s2_latent, p, num_iter=None, **kwargs):
        nfe = int(num_iter) if num_iter else self.num_inference_steps
        return self.sample(pixel_values, s2_latent, p, nfe)


class ActionHeadHVLAFlow(nn.Module):
    """HVLA S1 policy, flow decoder. Same interface as ActionHeadHVLA: pixels + the pooled S2 prefix latent."""

    pool_prefix = staticmethod(pool_prefix)

    def __init__(self, hidden_dim=896, action_dim=7, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = RecurrentConfigInternal(hidden_dim=hidden_dim, action_dim=action_dim)
        elif isinstance(cfg, dict):
            cfg = RecurrentConfigInternal(**cfg)
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.context_layer = getattr(cfg, "s1_context_layer", 23)
        self.model = S1HVLAFlow(cfg)

    def _proprio(self, B, proprio, proprio_projector):
        return self.model.proprio_token(proprio_projector(proprio.reshape(B, -1).to(torch.bfloat16)).unsqueeze(1))

    def training_loss(self, actions, pixel_values, s2_latent, proprio=None, proprio_projector=None, **kwargs):
        p = self._proprio(pixel_values.shape[0], proprio, proprio_projector)
        return self.model.loss(actions, pixel_values, s2_latent, p)

    def forward(self, pixel_values, s2_latent, p, num_iter=None, **kwargs):
        return self.model(pixel_values, s2_latent, p, num_iter=num_iter, **kwargs)

    def predict_action(self, pixel_values, s2_latent, proprio=None, proprio_projector=None,
                       phase="Inference", num_iter=None, **kwargs):
        p = self._proprio(pixel_values.shape[0], proprio, proprio_projector)
        nfe = int(num_iter) if num_iter else self.model.num_inference_steps
        return self.model.sample(pixel_values, s2_latent, p, nfe)
