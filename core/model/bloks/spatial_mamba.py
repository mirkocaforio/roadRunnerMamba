# -----------------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# -----------------------------------------------------------------------------------
# VMamba: Visual State Space Model
# Copyright (c) 2024 MzeroMiko
# -----------------------------------------------------------------------------------
# Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion
# Modified by Chaodong Xiao
# -----------------------------------------------------------------------------------

import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from .common import MLP
from .einFFT import EinFFT
from .stem import Stem
from .downsampling import DownSampling
from .prescanFGmask import PreScanFGMask

try:
    from ...utility.selective_scan import (
        selective_scan_state_flop_jit,
        selective_scan_fn,
    )
except:
    from utility.selective_scan import selective_scan_state_flop_jit, selective_scan_fn

try:
    from ...kernels.dwconv2d.Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None

"""
STAGE_CONFIGS = {
    0: dict(fg_k_ratio=0.50, fg_scale=2.0, mixer_type='einfft'),    # Stage 1 Local
    1: dict(fg_k_ratio=0.25, fg_scale=2.0, mixer_type='hybrid'),    # Stage 2 Mix
    2: dict(fg_k_ratio=0.125, fg_scale=2.0, mixer_type='hybrid'),   # Stage 3 Mix
    3: dict(fg_k_ratio=0.05, fg_scale=2.0, mixer_type='mlp'),       # Stage 4 Global
}
"""

STAGE_CONFIGS = {
    0: dict(fg_k_ratio=0.05, fg_scale=2.0, mixer_type="mlp"),  # Stage 1 Local
    1: dict(fg_k_ratio=0.05, fg_scale=2.0, mixer_type="mlp"),  # Stage 2 Mix
    2: dict(fg_k_ratio=0.05, fg_scale=2.0, mixer_type="mlp"),  # Stage 3 Mix
    3: dict(fg_k_ratio=0.05, fg_scale=2.0, mixer_type="mlp"),  # Stage 4 Global
}


class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.kernel_3 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

        # Buffer per il kernel fuso (eval). None = non ancora calcolato / invalidato.
        self.register_buffer("_merge_weight", None)

    def train(self, mode: bool = True):
        """Invalida il kernel fuso ogni volta che si torna in training."""
        super().train(mode)
        self._merge_weight = None
        return self

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode="replicate")

    def _build_merge_weight(self, device):
        """Costruisce il kernel 11x11 fuso dai tre kernel 3x3 + alpha."""
        w = torch.zeros((self.dim, 1, 11, 11), device=device)

        # kernel_3: dilation=1 → zona centrale [4:7, 4:7]
        w[:, :, 4:7, 4:7] = self.alpha[0] * self.kernel_3

        # kernel_3_1: dilation=3 → posizioni distanziate di 3
        a1 = self.alpha[1]
        rows_1 = [2, 5, 8]
        cols_1 = [2, 5, 8]
        for i, ri in enumerate(rows_1):
            for j, ci in enumerate(cols_1):
                w[:, :, ri : ri + 1, ci : ci + 1] += (
                    a1 * self.kernel_3_1[:, :, i : i + 1, j : j + 1]
                )

        # kernel_3_2: dilation=5 → posizioni distanziate di 5
        a2 = self.alpha[2]
        rows_2 = [0, 5, 10]
        cols_2 = [0, 5, 10]
        for i, ri in enumerate(rows_2):
            for j, ci in enumerate(cols_2):
                w[:, :, ri : ri + 1, ci : ci + 1] += (
                    a2 * self.kernel_3_2[:, :, i : i + 1, j : j + 1]
                )

        return w

    def forward(self, h):
        if self.training:
            # Tre conv depthwise separate (necessario per il gradiente su alpha e sui kernel).
            # Unico padding al massimo necessario (5,5,5,5) per evitare tre allocazioni.
            h_pad5 = self.padding(h, (5, 5, 5, 5))
            h1 = F.conv2d(
                h_pad5[:, :, 4:-4, 4:-4],
                self.kernel_3,
                padding=0,
                dilation=1,
                groups=self.dim,
            )
            h2 = F.conv2d(
                h_pad5[:, :, 2:-2, 2:-2],
                self.kernel_3_1,
                padding=0,
                dilation=3,
                groups=self.dim,
            )
            h3 = F.conv2d(
                h_pad5, self.kernel_3_2, padding=0, dilation=5, groups=self.dim
            )
            return self.alpha[0] * h1 + self.alpha[1] * h2 + self.alpha[2] * h3

        else:
            # Ricostruisce il kernel fuso solo se non è ancora disponibile.
            # NOTE: il centro [5,5] accumula contributi da tutti e tre i kernel
            # (dilation=1 mette kernel_3[1,1] in [5,5]; dilation=3 mette kernel_3_1[1,1]
            # in [5,5]; dilation=5 mette kernel_3_2[1,1] in [5,5]).
            # Questo è intenzionale: il pixel centrale riceve la somma pesata dei centri.
            if self._merge_weight is None:
                self._merge_weight = self._build_merge_weight(h.device)

            # Il kernel CUDA custom dwconv2d è fp32-only: eseguilo fuori autocast e
            # ripristina il dtype d'ingresso (bf16 sotto AMP).
            in_dtype = h.dtype
            with torch.autocast(device_type=h.device.type, enabled=False):
                out = DepthwiseFunction.apply(
                    h.float(), self._merge_weight.float(), None, 11 // 2, 11 // 2, False
                )
            return out.to(in_dtype)


class StructureAwareSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        use_fg_mask=True,
        fg_k_ratio=0.5,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fg_mask = use_fg_mask

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # Inizializzazione diretta come Parameter, senza creare+cancellare un Linear
        self.x_proj_weight = nn.Parameter(
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            ).weight
        )

        dt_proj = self.dt_init(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
            **factory_kwargs,
        )
        self.dt_projs_weight = nn.Parameter(dt_proj.weight)
        self.dt_projs_bias = nn.Parameter(dt_proj.bias)

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        self.state_fusion = StateFusion(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        if self.use_fg_mask:
            self.fg_mask = PreScanFGMask(
                d_inner=self.d_inner,
                k_ratio=fg_k_ratio,
            )
        else:
            self.fg_mask = None

    # ------------------------------------------------------------------
    # Metodi statici di inizializzazione
    # ------------------------------------------------------------------

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="he_normal",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=True,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        # Il default di prima era random
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "he_normal":
            nn.init.kaiming_normal_(dt_proj.weight, a=dt_init_std)
            # nn.init.kaiming_normal_(dt_proj.weight, nonlinearity='relu')
        elif dt_init == "he_uniform":
            nn.init.kaiming_uniform_(dt_proj.weight, a=dt_init_std)
            # nn.init.kaiming_uniform_(dt_proj.weight, nonlinearity='relu')
        elif dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
            dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
            dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError(f"dt_init='{dt_init}' non supportato.")

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        # FIX: `or "constant"` era sempre True → branch elif mai raggiunto
        if init in ("random", "constant", "he_normal", "he_uniform"):
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = nn.Parameter(torch.log(A))
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError(f"A_log init='{init}' non supportato.")
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        # FIX: `or "constant"` / `or "zero"` erano sempre True
        if init in ("random", "constant", "he_normal", "he_uniform"):
            D = nn.Parameter(torch.ones(d_inner, device=device))
            D._no_weight_decay = True
        elif init in ("simple", "zero"):
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError(f"D init='{init}' non supportato.")
        return D

    # ------------------------------------------------------------------

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        # [B, C, L]
        xs = x.reshape(B, C, L)

        # Proiezioni come matmul diretto nel layout [B, C, L]: il peso 2D fa broadcast sul
        # batch, evitando i doppi permute (stessa GEMM cuBLAS di F.linear, autocast-safe).
        # x_proj_weight: [rank+2S, C]  →  x_dbl: [B, rank+2S, L]
        x_dbl = torch.matmul(self.x_proj_weight, xs)  # [B, rank+2S, L]

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1
        )

        # dt_projs_weight: [C, rank]  →  dts: [B, C, L]
        dts = torch.matmul(self.dt_projs_weight, dts)  # [B, C, L]

        # FG mask: amplifica delta-t per i token foreground
        if self.fg_mask is not None:
            dts = dts * self.fg_mask(xs)

        As = -torch.exp(self.A_logs)
        dts = dts.contiguous()

        h = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            None,
            z=None,
            delta_bias=self.dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        # rearrange sostituito con view/reshape: evita il parsing della stringa einops
        # h: [B, C, 1, L] → [B, C, H, W]
        h = h.squeeze(2).reshape(B, C, H, W)
        h = self.state_fusion(h)
        h = h.reshape(B, C, L)

        y = h * Cs
        y = y + xs * self.Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # [B, H, W, C] → [B, C, H, W]  (un solo permute+contiguous)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))

        y = self.ssm(x)

        # [B, C, L] → [B, H, W, C]
        y = y.permute(0, 2, 1).reshape(B, H, W, self.d_inner)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class HybridMixer(nn.Module):
    """
    EinFFT + MLP residuale leggera (1 layer, hidden=dim).

    Usato negli stage Mix (2 e 3) per recuperare la capacità
    pointwise che EinFFT da solo perde rispetto alla MLP originale.
    """

    def __init__(self, dim: int, mlp_ratio: float = 1.0, mlp_drop: float = 0.0):
        super().__init__()
        self.fft = EinFFT(dim=dim)
        # FIX: hidden_features usava mlp_drop (float 0-1) come ratio → hidden=0 con drop=0
        self.mlp = MLP(
            in_features=dim,
            out_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=mlp_drop,
            depth=1,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, N, C]
        return self.fft(x, H=H, W=W) + self.mlp(x)


class CPE(nn.Module):
    """Conditional Positional Encoding: depthwise conv 3×3 con residual."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        # Un solo permute di andata e uno di ritorno, isolati qui
        return x + self.dw(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SpatialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        stage_idx: int = 0,
        use_fg_mask: bool = True,
        fg_k_ratio: float = None,
        fg_scale: float = 2.0,
        mixer_type: str = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # NOTE: per ora si usa sempre la config dello stage 3 (Global) per testare
        # l'effetto della maschera foreground. Ripristinare stage_idx quando pronto, al posto di 3
        # cfg = STAGE_CONFIGS.get(3, STAGE_CONFIGS[0])
        cfg = STAGE_CONFIGS.get(stage_idx, STAGE_CONFIGS[0])
        resolved_k_ratio = fg_k_ratio if fg_k_ratio is not None else cfg["fg_k_ratio"]
        resolved_mixer = mixer_type if mixer_type is not None else cfg["mixer_type"]

        self.cpe1 = CPE(hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = StructureAwareSSM(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            dt_init=dt_init,
            use_fg_mask=use_fg_mask,
            fg_k_ratio=resolved_k_ratio,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)

        self.cpe2 = CPE(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)

        if resolved_mixer == "einfft":
            self.mixer = EinFFT(dim=hidden_dim)
        elif resolved_mixer == "hybrid":
            # FIX: passa mlp_ratio separato da mlp_drop
            self.mixer = HybridMixer(
                dim=hidden_dim, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop_rate
            )
        elif resolved_mixer == "mlp":
            self.mixer = MLP(
                in_features=hidden_dim,
                out_features=hidden_dim,
                hidden_features=int(hidden_dim * mlp_ratio),
                dropout=mlp_drop_rate,
                depth=1,
            )
        else:
            raise ValueError(
                f"mixer_type '{resolved_mixer}' non riconosciuto. "
                f"Scegli tra: einfft, hybrid, mlp"
            )

        self._mixer_type = resolved_mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # CPE 1 + SSM
        x = self.cpe1(x)
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))

        # CPE 2 + mixer
        x = self.cpe2(x)
        x_ln = self.ln_2(x)

        if self._mixer_type == "mlp":
            # MLP opera su [B, H, W, C] direttamente
            mixer_out = self.mixer(x_ln)
        else:
            # EinFFT e HybridMixer operano su [B, N, C]
            mixer_out = self.mixer(x_ln.reshape(B, H * W, C), H=H, W=W)
            mixer_out = mixer_out.reshape(B, H, W, C)

        return x + self.drop_path(mixer_out)
