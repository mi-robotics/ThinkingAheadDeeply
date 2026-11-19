import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recurrent_vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from dataclasses import dataclass


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm * self.weight).to(dtype)


@dataclass
class RecurrentConfigInternal:
    hidden_dim: int = 896
    num_heads: int = 8
    # Inherited RD-VLA names: for HVLA only their LENGTHS matter (number of prelude / recurrent-core / coda
    # layers); the values were VLM layer indices in RD-VLA. e.g. [11], [23, 23], [23] = 1 + 2 + 1 layers.
    prelude_vlm_layers: tuple = ()
    recurrent_vlm_layers: tuple = (6, 23)
    coda_vlm_layers: tuple = ()
    action_chunk_len: int = 8
    action_dim: int = 7
    mean_recurrence: int = 12
    backprop_depth: int = 8
    random_iterations: bool = True
    init_std: float = 0.632
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    # Diagnostic B (rectified-recurrence project): RMSNorm the prelude output before the adapter so it cannot
    # swamp the normalised recurrent state. Off by default = the authors' architecture.
    prelude_out_norm: bool = False
    # Train every gradient-tracked iterate rather than only the final one. finetune.py already weights such a list
    # 1...n, so later iterates dominate; the released heads' weak-k=1 / strong-k=12 profile is what this should give.
    deep_supervision: bool = False
    # Scale of the S_0 noise alone (0 = fall back to init_std, which also sets the action-query init). A larger S_0
    # is a "blank workspace" one iteration cannot clean up, so refinement has to spread over more iterations.
    state_init_std: float = 0.0
    # --- HVLA S1 options. They live here because every loader rebuilds the head via
    # RecurrentConfigInternal(**saved_cfg) and the saved action_head_config carries these keys.
    s1_encoder_layers: int = 4
    s1_context_layer: int = 23
    s1_vit_model: str = "vit_small_patch14_dinov2.lvd142m"
    s1_vit_pretrained: bool = True
    s1_num_images: int = 2
    s1_s2_refresh_every: int = 8
    # Keep S1's DINOv2 tower at its pretrained weights (it is trainable by default).
    s1_freeze_vit: bool = False
    # --- flow decoder (type = hvla_flow). 4 untied layers: one NFE costs about one recurrent iteration.
    flow_num_layers: int = 4
    flow_num_inference_steps: int = 10
    # --- from-scratch student width (train_scratch_student.py). Width of the part a student owns: action
    # queries, prelude, adapter, recurrent core and the state S_k. 0 = hidden_dim (every trained head).
    # When set, those layers read the 896-wide context through ctx_dim = hidden_dim key/value projections, and a
    # learned state_proj (loop_dim -> hidden_dim) maps S_k into the width the (teacher's, frozen) coda decodes from.
    loop_dim: int = 0

    @property
    def weight_std(self) -> float:
        return math.sqrt(2.0 / (5.0 * self.hidden_dim))

    @property
    def output_std(self) -> float:
        return self.weight_std / math.sqrt(self.mean_recurrence * len(self.recurrent_vlm_layers))


class RecurrentLayer(nn.Module):
    """Recurrent layer: self-attention -> cross-attention (with gating) -> SwiGLU FFN."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, eps: float = 1e-6, rope_base: float = 10000.0,
                 ctx_dim: int = None):
        super().__init__()
        ctx_dim = ctx_dim or hidden_dim          # width of the context it cross-attends to (VLM features, proprio)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = RMSNorm(hidden_dim, eps)
        self.q_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_self = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)

        self.norm2 = RMSNorm(hidden_dim, eps)
        self.q_cross = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_latents = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.v_latents = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.k_vision = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.v_vision = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.o_cross = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

        self.norm3 = RMSNorm(hidden_dim, eps)
        self.ffn_gate = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, latent_tokens: torch.Tensor, vision_tokens: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        def reshape(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Self-Attention
        residual = x
        x_n = self.norm1(x)
        q_s = reshape(self.q_self(x_n), T)
        k_s = reshape(self.k_self(x_n), T)
        v_s = reshape(self.v_self(x_n), T)
        cos, sin = self.rope(T, x.device, x.dtype)
        q_s, k_s = apply_rope(q_s, k_s, cos, sin)
        scale = self.head_dim ** -0.5
        attn_s = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
        attn_s = F.softmax(attn_s, dim=-1)
        out_s = torch.matmul(attn_s, v_s)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_self(out_s)

        # 2. Cross-Attention
        context_latents = torch.cat([latent_tokens, p], dim=1)
        K_latents_len = context_latents.size(1)
        K_vision_len = vision_tokens.size(1)
        residual = x
        x_n = self.norm2(x)
        q_c = reshape(self.q_cross(x_n), T)
        k_lat = reshape(self.k_latents(context_latents), K_latents_len)
        v_lat = reshape(self.v_latents(context_latents), K_latents_len)
        k_vis = reshape(self.k_vision(vision_tokens), K_vision_len)
        v_vis = reshape(self.v_vision(vision_tokens), K_vision_len)
        attn_latents = torch.matmul(q_c, k_lat.transpose(-2, -1))
        attn_vision = torch.matmul(q_c, k_vis.transpose(-2, -1)) * torch.tanh(self.gate)
        attn_weights = F.softmax(torch.cat([attn_latents, attn_vision], dim=-1) * scale, dim=-1)
        v_combined = torch.cat([v_lat, v_vis], dim=2)
        out_c = torch.matmul(attn_weights, v_combined)
        out_c = out_c.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_cross(out_c)

        # 3. SwiGLU FFN
        residual = x
        x_n = self.norm3(x)
        x = residual + self.ffn_down(F.silu(self.ffn_gate(x_n)) * self.ffn_up(x_n))

        return x


class VLARecurrent(nn.Module):
    """Parameters of the recurrent-depth decoder: prelude -> weight-tied recurrent core (iterated) -> coda.

    The forward pass over a context lives in hvla_recurrent.S1HVLARecurrent; this class owns the layers,
    their initialisation, the S_0 noise and the training-time iteration-count sampler.
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = RecurrentConfigInternal(**cfg)
        self.cfg = cfg
        dim = cfg.hidden_dim

        self.prelude_vlm_layers = list(cfg.prelude_vlm_layers)
        self.recurrent_vlm_layers = list(cfg.recurrent_vlm_layers)
        self.coda_vlm_layers = list(cfg.coda_vlm_layers)

        # loop_dim (ablation 3): width of the queries, prelude, adapter, recurrent core and state. Equal to dim for
        # every existing head, in which case nothing below differs from the original architecture.
        L = self.loop_dim = getattr(cfg, "loop_dim", 0) or dim
        self.action_queries = nn.Parameter(torch.randn(cfg.action_chunk_len, L) * cfg.init_std)

        if self.prelude_vlm_layers:
            self.prelude = nn.ModuleList([
                RecurrentLayer(L, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base, ctx_dim=dim)
                for _ in self.prelude_vlm_layers
            ])

        if getattr(cfg, "prelude_out_norm", False):
            self.prelude_out_norm = RMSNorm(L, cfg.rms_norm_eps)

        self.adapter = nn.Linear(L * 2, L, bias=False)
        self.adapter_norm = RMSNorm(L, cfg.rms_norm_eps)
        self.gamma_adapt = nn.Parameter(torch.ones(1))
        self.recurrent = nn.ModuleList([
            RecurrentLayer(L, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base, ctx_dim=dim)
            for _ in self.recurrent_vlm_layers
        ])
        self.recurrent_norm = RMSNorm(L, cfg.rms_norm_eps)
        # S_k (width L) -> the width the coda decodes from. Present whenever loop_dim is SET (a from-scratch student,
        # including loop_dim == dim, so every point of the width sweep has the same structure); absent when it is 0, so
        # the parameter names of every existing checkpoint are unchanged.
        if getattr(cfg, "loop_dim", 0):
            self.state_proj = nn.Linear(L, dim, bias=False)

        if self.coda_vlm_layers:
            self.coda = nn.ModuleList([
                RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
                for _ in self.coda_vlm_layers
            ])

        self.output_norm = RMSNorm(dim, cfg.rms_norm_eps)
        self.output_proj = nn.Linear(dim, cfg.action_dim)
        self.gamma_init = nn.Parameter(torch.ones(1))
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        weight_std = cfg.weight_std
        output_std = cfg.output_std
        # Layers that live at loop_dim get the same rule evaluated at their own width (identical when loop_dim == dim).
        L = self.loop_dim
        loop_weight_std = math.sqrt(2.0 / (5.0 * L))
        loop_output_std = loop_weight_std / math.sqrt(cfg.mean_recurrence * len(cfg.recurrent_vlm_layers))
        loop_parts = ("prelude.", "adapter", "recurrent.", "state_proj")
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_output = any(x in name for x in ['o_proj', 'ffn_down', 'output_proj'])
                if name.startswith(loop_parts):
                    std = loop_output_std if is_output else loop_weight_std
                else:
                    std = output_std if is_output else weight_std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.action_queries, mean=0.0, std=cfg.init_std, a=-3*cfg.init_std, b=3*cfg.init_std)

    def init_state(self, B: int, device, dtype) -> torch.Tensor:
        base_std = getattr(self.cfg, "state_init_std", 0.0) or self.cfg.init_std
        std = (self.gamma_init * base_std).item()
        state = torch.empty(B, self.cfg.action_chunk_len, self.loop_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3*std, b=3*std)
        return state

    def sample_iterations(self) -> int:
        r_mean = self.cfg.mean_recurrence
        tau = torch.normal(mean=math.log(r_mean) - 0.125, std=0.5, size=(1,))
        lam = torch.exp(tau).clamp(max=100)
        return max(1, min(torch.poisson(lam).int().item() + 1, 64))
