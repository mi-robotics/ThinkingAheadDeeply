"""HVLA with the recurrent-depth decoder (config type `hvla`).

S1 context (hvla_context.S1ContextMixin) -> prelude -> weight-tied recurrent core iterated k times
-> coda -> an action chunk. Training samples k (Poisson-lognormal, mean `mean_recurrence`) and
backpropagates through the last `backprop_depth` iterations; evaluation fixes k
(`--recurrent_num_iter`) or stops on convergence.

The names `build_prelude`, `init_state`, `adapter`, `_run_one_iteration` and `_get_output` are the
hooks finetune.py's health / depth probes and the rectification scripts (rectify_*.py,
frozen_backbone.py) call. Each recomputes the context from (pixel_values, s2_latent, p).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from recurrent_vla.models.recurrent_core import RecurrentConfigInternal, VLARecurrent
from recurrent_vla.models.hvla_context import S1ContextMixin, pool_prefix


class S1HVLARecurrent(S1ContextMixin, VLARecurrent):
    """Recurrent decoder over the S1 context."""

    def __init__(self, cfg):
        VLARecurrent.__init__(self, cfg)      # decoder parameters + their init
        self._init_s1_context(cfg)            # after the decoder init, so the pretrained ViT is untouched

    # --- decoder stages over a built context ----------------------------------------------
    def build_prelude_ctx(self, ctx_v, ctx_l, p, dtype):
        B = ctx_v.size(0)
        x = self.action_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
        if self.prelude_vlm_layers:
            for layer in self.prelude:
                x = layer(x, ctx_l, ctx_v, p)
        if getattr(self.cfg, "prelude_out_norm", False):
            x = self.prelude_out_norm(x)
        return x

    def iterate(self, state, prelude_out, ctx_v, ctx_l, p):
        x = self.adapter(torch.cat([state, prelude_out], dim=-1))
        x = self.adapter_norm(self.gamma_adapt * x)
        for layer in self.recurrent:
            x = layer(x, ctx_l, ctx_v, p)
        return self.recurrent_norm(x)

    def decode(self, state, ctx_v, ctx_l, p):
        x = self.state_proj(state) if hasattr(self, "state_proj") else state   # loop_dim -> hidden_dim (ablation 3)
        if self.coda_vlm_layers:
            for layer in self.coda:
                x = layer(x, ctx_l, ctx_v, p)
        return self.output_proj(self.output_norm(x))

    # --- forward ---------------------------------------------------------------------------
    def forward(self, pixel_values, s2_latent, p, num_iter=None, convergence_strategy=None,
                kl_thresh=0.001, cos_thresh=0.999, max_iter=32, **kwargs):
        ctx_v, ctx_l = self.build_context(pixel_values, s2_latent, p)
        dtype, device = ctx_v.dtype, ctx_v.device
        B = ctx_v.size(0)

        prelude_out = self.build_prelude_ctx(ctx_v, ctx_l, p, dtype)
        state = self.init_state(B, device, dtype)

        if convergence_strategy in ("kl_divergence", "cosine_similarity") and not self.training:
            prev, actual, final = None, 0, None
            with torch.no_grad():
                for it in range(max_iter):
                    state = self.iterate(state, prelude_out, ctx_v, ctx_l, p)
                    actual = it + 1
                    cur = self.decode(state, ctx_v, ctx_l, p)
                    if prev is not None:
                        if convergence_strategy == "cosine_similarity":
                            cs = torch.nn.functional.cosine_similarity(prev.flatten(), cur.flatten(), dim=0).item()
                            final = 1 - cs
                            if cs > cos_thresh:
                                break
                        else:
                            mse = torch.mean((cur - prev) ** 2).item()
                            final = mse
                            if mse < kl_thresh:
                                break
                    prev = cur
            return self.decode(state, ctx_v, ctx_l, p), actual, final

        total = num_iter if num_iter is not None else (
            self.sample_iterations() if (self.training and self.cfg.random_iterations)
            else self.cfg.mean_recurrence)
        k = self.cfg.backprop_depth
        n_no_grad = max(0, total - k)
        if n_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_no_grad):
                    state = self.iterate(state, prelude_out, ctx_v, ctx_l, p)

        if self.training and getattr(self.cfg, "deep_supervision", False):
            outs = []
            for _ in range(min(k, total)):
                state = self.iterate(state, prelude_out, ctx_v, ctx_l, p)
                outs.append(self.decode(state, ctx_v, ctx_l, p))
            return outs

        for _ in range(min(k, total)):
            state = self.iterate(state, prelude_out, ctx_v, ctx_l, p)
        return self.decode(state, ctx_v, ctx_l, p)

    # --- hooks for the probes and rectification (context recomputed per call, never cached) --
    def build_prelude(self, pixel_values, s2_latent, p):
        ctx_v, ctx_l = self.build_context(pixel_values, s2_latent, p)
        return self.build_prelude_ctx(ctx_v, ctx_l, p, ctx_v.dtype)

    def _run_one_iteration(self, state, prelude_out, pixel_values, s2_latent, p):
        ctx_v, ctx_l = self.build_context(pixel_values, s2_latent, p)
        return self.iterate(state, prelude_out, ctx_v, ctx_l, p)

    def _get_output(self, state, pixel_values, s2_latent, p):
        ctx_v, ctx_l = self.build_context(pixel_values, s2_latent, p)
        return self.decode(state, ctx_v, ctx_l, p)


class ActionHeadHVLA(nn.Module):
    """HVLA S1 policy, recurrent decoder. Consumes pixels + the pooled S2 prefix latent; never touches the LLM."""

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
        self.model = S1HVLARecurrent(cfg)

    def forward(self, pixel_values, s2_latent, p, num_iter=None, **kwargs):
        return self.model(pixel_values, s2_latent, p, num_iter=num_iter, **kwargs)

    def predict_action(self, pixel_values, s2_latent, proprio=None, proprio_projector=None,
                       phase="Inference", num_iter=None, convergence_strategy=None,
                       kl_thresh=0.001, cos_thresh=0.999, max_iter=32, **kwargs):
        B = pixel_values.shape[0]
        p = self.model.proprio_token(proprio_projector(proprio.reshape(B, -1).to(torch.bfloat16)).unsqueeze(1))
        return self.model(pixel_values, s2_latent, p, num_iter=num_iter,
                          convergence_strategy=convergence_strategy, kl_thresh=kl_thresh,
                          cos_thresh=cos_thresh, max_iter=max_iter)
