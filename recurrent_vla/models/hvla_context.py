"""The HVLA S1 observation stack, shared by every HVLA head (recurrent and flow).

Split
-----
  S2 (slow)   the VLM. One forward over images + the language prompt; its PREFIX (the image
              patch tokens and the instruction's own text tokens) is mean-pooled at one layer
              into a single [B, D] latent (`pool_prefix`). This is what carries the instruction.
  S1 (fast)   a DINOv2 ViT-S over the frames + proprio + the (possibly stale) S2 latent ->
              observation encoder -> a decoder (recurrent loop or flow), see hvla_recurrent.py
              and hvla_flow.py.

`S1ContextMixin` builds everything up to the decoder: the vision tokens (current-frame patches
and/or image-history CLS tokens), the projected S2 token, and the observation encoder that fuses
them with proprio into the two context streams every decoder layer cross-attends to.

Vision-token modes (constants.IMAGE_HISTORY > 1 and RDVLA_HIST_VIT; both must match at eval)
--------------------------------------------------------------------------------------------
  no history     256 patch tokens per camera of the current frame (IMAGE_HISTORY = 1).
  shared         + one CLS token per (frame, camera) of the last H frames from the same ViT,
                 computed WITHOUT gradient. Default when IMAGE_HISTORY > 1.
  shared_grad    as `shared`, with gradient through the history frames (gradient checkpointing).
  separate       history CLS from a second, separately trained ViT-S.
  memhvla        Zeng et al.'s double encoder (Revisiting Open-Loop Execution, arXiv 2608.15938;
                 diffusion-policy-experiments): NO patch tokens. A long-range ViT (self.vit) gives
                 the CLS of all H frames, a separately-weighted short-range ViT (self.short_vit)
                 the CLS of the last 2; both trained with gradient through every frame; the
                 short-range tokens are zeroed per sample w.p. RDVLA_SHORT_DROPOUT (0.3) in
                 training. Pair with RDVLA_ENCODER_LR_SCALE=0.1 (finetune.py) -- at the full
                 learning rate gradient through history drifts the ViT and success collapses.

Input contract
--------------
`pixel_values` is channel-stacked per image as [DINOv2(3) | SigLIP(3)], the DINOv2 block already
ImageNet-normalised, so S1 slices channels 6i:6i+3. History frames arrive separately as uint8
[B, H, n, h, w, 3] through the `_hist_images` side channel (set by finetune.py / openvla_utils.py
before each forward) and are normalised here.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from recurrent_vla.models.recurrent_core import RMSNorm
from recurrent_vla.constants import ACTION_TOKEN_BEGIN_IDX


def dinov2_views(pixel_values: torch.Tensor, num_images: int) -> torch.Tensor:
    """[B, 6*n, H, W] channel-stacked -> [B*n, 3, H, W] of DINOv2-normalised views (view-major)."""
    views = [pixel_values[:, 6 * i: 6 * i + 3] for i in range(num_images)]
    return torch.cat(views, dim=0)


def pool_prefix(hidden_states, layer: int, labels: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Mean-pool the VLM prefix at `layer` -> [B, D], the S2 latent.

    `hidden_states` is the raw per-layer sequence from the VLM, each [B, T, D] with
    T = num_patches + L, laid out as

        index 0                      the prompt's first token
        indices 1 .. num_patches     the vision patches
        indices num_patches+1 ..     the rest of the prompt, then the action placeholders,
                                     the stop token, and any right padding

    `labels` is [B, L] over the un-spliced sequence, with action positions carrying token ids
    above ACTION_TOKEN_BEGIN_IDX. The first such position marks where the prefix ends, so the
    prefix is indices 0 .. num_patches + a0 - 1. The bound is per sample because the collator
    right-pads variable-length instructions; padding is excluded by the same bound.

    Pooling the prefix (not an assembled patch + action-token stack) is what keeps the
    instruction tokens in the latent. Attention is causal, so the action placeholders after the
    prefix cannot change its hidden states.
    """
    item = hidden_states[layer]
    B, T, _ = item.shape
    lab = labels.to(item.device)
    is_act = lab > ACTION_TOKEN_BEGIN_IDX
    if not bool(is_act.any(dim=1).all()):
        # Silently pooling only the patches would look like a working run with the
        # instruction removed.
        raise ValueError(
            "pool_prefix: a sample has no action-token labels, so the end of the prefix "
            "cannot be located. Check that `labels` is the un-spliced label tensor."
        )
    a0 = torch.argmax(is_act.to(torch.int8), dim=1)          # [B] first action position
    prefix_len = (num_patches + a0).clamp(max=T)             # [B]
    pos = torch.arange(T, device=item.device).unsqueeze(0)   # [1, T]
    mask = (pos < prefix_len.unsqueeze(1)).unsqueeze(-1).to(item.dtype)
    return (item * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class S1ObservationEncoder(nn.Module):
    """Fuses the vision tokens, the S2 token and proprio into one context.

    Pre-norm so it composes with the RMSNorm-based decoder layers without a scale mismatch at init.
    """

    def __init__(self, dim: int, num_heads: int, num_layers: int, eps: float = 1e-6, ff_mult: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = RMSNorm(dim, eps)

    def forward(self, h_v: torch.Tensor, s2: torch.Tensor, p: torch.Tensor):
        """h_v [B, N_vis, D], s2 [B, 1, D], p [B, 1, D] -> (ctx_vision [B, N_vis, D], ctx_latent [B, 2, D])."""
        ctx = self.norm(self.encoder(torch.cat([h_v, s2, p], dim=1)))
        n_vis = h_v.size(1)
        return ctx[:, :n_vis], ctx[:, n_vis:]


def _make_vit(cfg):
    import timm

    return timm.create_model(
        getattr(cfg, "s1_vit_model", "vit_small_patch14_dinov2.lvd142m"),
        pretrained=bool(getattr(cfg, "s1_vit_pretrained", True)),
        num_classes=0,
        img_size=224,
    )


class S1ContextMixin:
    """Mixed into an nn.Module decoder; call `_init_s1_context(cfg)` from its __init__.

    Attribute names (vit, short_vit, hist_vit, _hist_images, image_history) are read by
    finetune.py (encoder LR group, history side channel) and openvla_utils.py.
    """

    def _init_s1_context(self, cfg):
        from recurrent_vla.constants import IMAGE_HISTORY

        dim = cfg.hidden_dim
        self.num_images = getattr(cfg, "s1_num_images", 2)
        self.vit = _make_vit(cfg)
        self.freeze_vit = bool(getattr(cfg, "s1_freeze_vit", False))
        if self.freeze_vit:
            for q in self.vit.parameters():
                q.requires_grad = False
            self.vit.eval()
        vit_dim = self.vit.embed_dim
        # The S2 latent arrives in the VLM's width; a learned projection lets S1 reinterpret it.
        self.s2_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.s2_norm = RMSNorm(dim, cfg.rms_norm_eps)

        self.image_history = IMAGE_HISTORY
        self.hist_vit_mode = os.environ.get("RDVLA_HIST_VIT", "shared") if IMAGE_HISTORY > 1 else "none"
        if self.hist_vit_mode not in ("none", "shared", "shared_grad", "separate", "memhvla"):
            raise ValueError(f"RDVLA_HIST_VIT={self.hist_vit_mode!r}: expected shared | shared_grad | separate | memhvla")
        if self.hist_vit_mode != "memhvla":
            self.vit_proj = nn.Linear(vit_dim, dim, bias=False)     # current-frame patch tokens
        if self.hist_vit_mode in ("shared_grad", "memhvla"):
            self.vit.set_grad_checkpointing(True)                   # many frames per sample with gradient
        if self.hist_vit_mode == "separate":
            self.hist_vit = _make_vit(cfg)
            if os.environ.get("RDVLA_HIST_VIT_GRAD_CKPT") == "1":
                self.hist_vit.set_grad_checkpointing(True)
        if self.hist_vit_mode == "memhvla":
            self.short_vit = _make_vit(cfg)
            self.short_vit.set_grad_checkpointing(True)
            self.short_range = 2
            self.short_dropout = float(os.environ.get("RDVLA_SHORT_DROPOUT", "0.3"))
            self.short_proj = nn.Linear(vit_dim, dim, bias=False)
            self.short_time = nn.Parameter(torch.zeros(self.short_range, dim))
            self.hist_range = nn.Parameter(torch.zeros(2, dim))        # 0 = long, 1 = short
            nn.init.normal_(self.short_time, std=0.02)
            nn.init.normal_(self.hist_range, std=0.02)
        if IMAGE_HISTORY > 1:
            self.hist_proj = nn.Linear(vit_dim, dim, bias=False)
            self.hist_time = nn.Parameter(torch.zeros(IMAGE_HISTORY, dim))
            self.hist_cam = nn.Parameter(torch.zeros(self.num_images, dim))
            nn.init.normal_(self.hist_time, std=0.02)
            nn.init.normal_(self.hist_cam, std=0.02)
            self.register_buffer("hist_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("hist_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self._hist_images = None      # [B, H, n, h, w, 3] uint8, set by the caller before a forward (side channel)
        self.obs_encoder = S1ObservationEncoder(
            dim=dim,
            num_heads=cfg.num_heads,
            num_layers=getattr(cfg, "s1_encoder_layers", 4),
            eps=cfg.rms_norm_eps,
        )
        # RDVLA_ACTION_HISTORY (data/action_history.py): actions t-H .. t-1-G -> a term added to the proprio token.
        # Last layer zero-initialised, so the history arm starts exactly at the no-history model.
        from recurrent_vla.constants import ACTION_DIM, ACTION_GAP, ACTION_HIST_DROP, ACTION_HISTORY
        self.action_history = ACTION_HISTORY
        if ACTION_HISTORY:
            self.act_hist_drop = ACTION_HIST_DROP
            self.act_hist_proj = nn.Sequential(nn.Linear((ACTION_HISTORY - ACTION_GAP) * ACTION_DIM, dim), nn.GELU(),
                                               nn.Linear(dim, dim))
            nn.init.zeros_(self.act_hist_proj[2].weight)
            nn.init.zeros_(self.act_hist_proj[2].bias)
        self._action_hist = None      # [B, H - G, ACTION_DIM] normalised actions, set by the caller (side channel)

    def proprio_token(self, p):
        """p [B, 1, D] (the projected proprio) + the action-history term when RDVLA_ACTION_HISTORY is on.
        Without a supplied history (e.g. a probe that does not set one) the token is returned unchanged."""
        if not self.action_history or self._action_hist is None:
            return p
        h = self.act_hist_proj(self._action_hist.to(p.device).flatten(1).to(self.act_hist_proj[0].weight.dtype))
        if self.training and self.act_hist_drop > 0:     # "actdrop": zero the whole term for a fraction of samples
            keep = (torch.rand(h.shape[0], 1, device=h.device) >= self.act_hist_drop).to(h.dtype)
            h = h * keep
        return p + h.to(p.dtype).unsqueeze(1)

    def train(self, mode: bool = True):
        """A frozen tower stays in eval mode when the parent goes to train()."""
        super().train(mode)
        if getattr(self, "freeze_vit", False):
            self.vit.eval()
        return self

    # --- context -------------------------------------------------------------------------
    def build_context(self, pixel_values, s2_latent, p):
        """-> (ctx_vision [B, N_vis, D], ctx_latent [B, 2, D])."""
        B = pixel_values.shape[0]
        views = dinov2_views(pixel_values, self.num_images)          # [B*n, 3, H, W]
        if self.hist_vit_mode == "memhvla":
            feats = self._memhvla_tokens(views, B)                   # [B, H*n + 2*n, D]
        else:
            feats = self.vit.forward_features(views)                 # [B*n, P(+cls), C]
            if feats.shape[1] % 2 == 1:                              # drop the cls token
                feats = feats[:, 1:]
            feats = self.vit_proj(feats.to(self.vit_proj.weight.dtype))
            P, D = feats.shape[1], feats.shape[2]
            feats = feats.view(self.num_images, B, P, D).permute(1, 0, 2, 3).reshape(B, self.num_images * P, D)
            if self.image_history > 1:
                feats = torch.cat([feats, self._history_tokens(views, B).to(feats.dtype)], dim=1)
        s2 = self.s2_norm(self.s2_proj(s2_latent.to(feats.dtype))).unsqueeze(1)   # [B, 1, D]
        return self.obs_encoder(feats, s2, p)

    def _history_pixels(self, views, B):
        """uint8 history [B, H, n, h, w, 3] -> ImageNet-normalised [B, H, n, 3, h', w'] at the views' size."""
        H, n = self.image_history, self.num_images
        x = self._hist_images.to(views.device)
        x = x.reshape(B, H, n, *x.shape[3:]).permute(0, 1, 2, 5, 3, 4).to(views.dtype) / 255.0
        if x.shape[-1] != views.shape[-1]:
            x = torch.nn.functional.interpolate(x.flatten(0, 2), size=views.shape[-2:], mode="bilinear",
                                                align_corners=False).view(B, H, n, 3, *views.shape[-2:])
        return (x - self.hist_mean.to(x.dtype)) / self.hist_std.to(x.dtype)

    def _history_tokens(self, views, B):
        """-> [B, H*n, D]: CLS of each (frame, camera). Without a supplied history (e.g. the depth/health
        probes) the current frame stands in for every slot -- the same as an episode's first step."""
        H, n = self.image_history, self.num_images
        enc = self.hist_vit if self.hist_vit_mode == "separate" else self.vit
        with_grad = self.hist_vit_mode in ("separate", "shared_grad")
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):  # "shared": no gradient via history
            if self._hist_images is None:
                cls = enc.forward_features(views)[:, 0]                                 # [n*B, C], view-major
                cls = cls.view(n, B, -1).permute(1, 0, 2).unsqueeze(1).expand(B, H, n, -1)
            else:
                x = self._history_pixels(views, B)
                cls = enc.forward_features(x.flatten(0, 2))[:, 0].view(B, H, n, -1)
        tok = self.hist_proj(cls.to(self.hist_proj.weight.dtype))                        # [B, H, n, D]
        tok = tok + self.hist_time[None, :, None] + self.hist_cam[None, None, :]
        return tok.reshape(B, H * n, -1)

    def _memhvla_tokens(self, views, B):
        """-> [B, H*n + 2*n, D]: long-range CLS of every (frame, camera), then short-range CLS of the last
        2 frames, each projected and tagged with time, camera and range embeddings."""
        H, n, S = self.image_history, self.num_images, self.short_range
        if self._hist_images is None:
            lc = self.vit.forward_features(views)[:, 0]
            sc = self.short_vit.forward_features(views)[:, 0]
            lc = lc.view(n, B, -1).permute(1, 0, 2).unsqueeze(1).expand(B, H, n, -1)
            sc = sc.view(n, B, -1).permute(1, 0, 2).unsqueeze(1).expand(B, S, n, -1)
        else:
            x = self._history_pixels(views, B)
            lc = self.vit.forward_features(x.flatten(0, 2))[:, 0].view(B, H, n, -1)
            sc = self.short_vit.forward_features(x[:, H - S:].flatten(0, 2))[:, 0].view(B, S, n, -1)
        lt = self.hist_proj(lc.to(self.hist_proj.weight.dtype))                           # [B, H, n, D]
        st = self.short_proj(sc.to(self.short_proj.weight.dtype))                         # [B, S, n, D]
        if self.training and self.short_dropout > 0:
            keep = (torch.rand(B, device=st.device) >= self.short_dropout).to(st.dtype)
            st = st * keep.view(B, 1, 1, 1)
        lt = lt + self.hist_time[None, :, None] + self.hist_cam[None, None, :] + self.hist_range[0]
        st = st + self.short_time[None, :, None] + self.hist_cam[None, None, :] + self.hist_range[1]
        return torch.cat([lt.reshape(B, H * n, -1), st.reshape(B, S * n, -1)], dim=1)
