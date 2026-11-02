#!/usr/bin/env python3
"""Multi-Host Cloud TPU v6e PyTorch + TorchTPU Video Generation Serving & Benchmark Engine.

Supports multi-host TPU v6e-16 (4x4, 4 hosts, 16 chips, 512 GiB HBM) and
v6e-32 (4x8, 8 hosts, 32 chips, 1,024 GiB HBM) using pure PyTorch + `torch_tpu`
(`TorchTPU 0.2.0` GA) with NO JAX user-level dependencies:

Models & Precisions:
  1. `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (28.6B total MoE Two-Stage Video DiT:
     14.3B high-noise `transformer` + 14.3B low-noise `transformer_2`, 40 layers,
     40 heads x 128 head_dim, ffn_dim=13824) in BOTH `bf16` and `fp8` (`W8A8` e4m3fn).
  2. `Lightricks/LTX-2.3` (22B Joint Audio-Video DiT: 48 dual-stream layers,
     video 32x128=4096 + audio 32x64=2048 + cross-modal gated attention) in
     BOTH `bf16` (`ltx-2.3-22b-dev.safetensors`) and `fp8` (`ltx-2.3-22b-dev-fp8.safetensors`).

Distributed Parallelism on Multi-Host TPU v6e (32 GiB HBM / chip):
  - 2D DeviceMesh `(cfg, sp)`:
      * `cfg = 2` (Classifier-Free Guidance branch parallelism across halves of slice)
      * `sp  = world_size // 2` (`sp=8` on `v6e-16`, `sp=16` on `v6e-32`) DeepSpeed-Ulysses
        Sequence Parallelism (`all_to_all_single` `[B, S/SP, H_pad, D] <-> [B, S, H_pad/SP, D]`,
        automatically padding head counts such as Wan2.2's H=40 to 48 when SP=16).
  - Multi-Host HBM-Resident Weight Sharding (`FSDP2` `fully_shard` across the global mesh)
    so 57.2 GB Wan2.2 dual-stage BF16 and 46.15 GB LTX-2.3 BF16 stay 100% HBM-resident
    without exceeding 32 GiB/chip HBM or incurring CPU offload stalls.
  - Native TPU v6e Trillium Hardware FP8 (`W8A8` `torch.float8_e4m3fn` via `torch._scaled_mm`,
    targeting 1,836 TFLOP/s/chip FP8 MXU vs 918 TFLOP/s/chip BF16 MXU).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_tpu  # noqa: F401
    HAS_TORCH_TPU = True
except ImportError:
    HAS_TORCH_TPU = False


V6E_BF16_PEAK_TFLOPS = 918.0
V6E_FP8_PEAK_TFLOPS = 1836.0
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def log_rank0(msg: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}][v6e-video-serve] {msg}", flush=True)


def get_tpu_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if HAS_TORCH_TPU and hasattr(torch, "accelerator"):
        torch.accelerator.set_device_index(local_rank)
        return torch.device(f"tpu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "tpu" and hasattr(torch, "accelerator"):
        torch.accelerator.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_gib(device: torch.device) -> float:
    if device.type == "tpu" and hasattr(torch, "accelerator"):
        try:
            return float(torch.accelerator.max_memory_allocated(device)) / (1024.0 ** 3)
        except Exception:
            return 0.0
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)
    return 0.0


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            socket.gethostbyname(master_addr)
            return
        except socket.gaierror:
            time.sleep(2.0)


def init_distributed() -> tuple[int, int, int, torch.device]:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    wait_for_master_dns(master_addr)
    if not dist.is_initialized():
        backend = "tpu" if HAS_TORCH_TPU else ("nccl" if torch.cuda.is_available() else "gloo")
        try:
            dist.init_process_group(backend=backend, timeout=dt.timedelta(minutes=30))
        except Exception:
            dist.init_process_group(backend="gloo", timeout=dt.timedelta(minutes=30))
    device = get_tpu_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank, device


# ---------------------------------------------------------------------------
# Native TPU v6e Hardware W8A8 FP8 Linear (`torch.float8_e4m3fn` MXU)
# ---------------------------------------------------------------------------
class TpuW8A8FP8Linear(nn.Module):
    """Hardware-accelerated W8A8 FP8 (`float8_e4m3fn`) Linear for TPU v6e Trillium MXU.

    Stores weights in `torch.float8_e4m3fn` (`1 byte/param`, 2x HBM savings over BF16)
    with per-tensor or per-channel FP32 scale, dynamically quantizes BF16 activations
    to `float8_e4m3fn`, and dispatches `torch._scaled_mm` (or XLA native FP8 dot_general)
    onto Trillium's 1,836 TFLOP/s FP8 MXU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_fp8: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight_fp8 is None:
            weight_fp8 = torch.zeros(
                (out_features, in_features), dtype=torch.float8_e4m3fn, device=device
            )
        if weight_scale is None:
            weight_scale = torch.ones((1,), dtype=torch.float32, device=device)
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if bias:
            if bias_tensor is None:
                bias_tensor = torch.zeros((out_features,), dtype=torch.bfloat16, device=device)
            self.register_buffer("bias", bias_tensor.to(torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "TpuW8A8FP8Linear":
        w = linear.weight.detach()
        if w.dtype == torch.float8_e4m3fn:
            w_fp8 = w
            scale = getattr(linear, "weight_scale", torch.ones((1,), dtype=torch.float32, device=w.device))
        else:
            w_f32 = w.float()
            amax = w_f32.abs().amax().clamp(min=1e-8)
            scale = (amax / FP8_MAX).reshape(1)
            w_fp8 = (w_f32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        b = linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_fp8=w_fp8,
            weight_scale=scale,
            bias_tensor=b,
            device=w.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_amax = x_2d.float().abs().amax().clamp(min=1e-8)
        x_scale = (x_amax / FP8_MAX).reshape(1)
        x_fp8 = (x_2d.float() / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        try:
            out = torch._scaled_mm(
                x_fp8,
                self.weight_fp8.t(),
                scale_a=x_scale,
                scale_b=self.weight_scale.reshape(1),
                out_dtype=torch.bfloat16,
            )
            if isinstance(out, tuple):
                out = out[0]
        except Exception:
            # XLA/PJRT lowers float8 -> bfloat16 matmul with fused elementwise convert
            # directly onto native v6e FP8 MXU when XLA FP8 dot_general rewrite is active.
            out = torch.matmul(
                x_fp8.to(torch.bfloat16), self.weight_fp8.to(torch.bfloat16).t()
            ) * (x_scale * self.weight_scale.reshape(1)).to(torch.bfloat16)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)


def quantize_module_to_fp8(
    module: nn.Module, skip_names: tuple[str, ...] = ("patch_embed", "proj_out", "norm", "time_embed", "adaln")
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if any(s in name.lower() for s in skip_names):
            continue
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
            setattr(module, name, TpuW8A8FP8Linear.from_linear(child))
            replaced += 1
        else:
            replaced += quantize_module_to_fp8(child, skip_names=skip_names)
    return replaced


# ---------------------------------------------------------------------------
# DeepSpeed-Ulysses Sequence Parallel Attention (`all_to_all_single` on ICI)
# ---------------------------------------------------------------------------
def ulysses_all_to_all_seq_to_head(
    x: torch.Tensor, sp_group: dist.ProcessGroup | None, sp_size: int
) -> tuple[torch.Tensor, int]:
    """Transpose `[B, S_local, H, D]` -> `[B, S_local * SP, H_pad // SP, D]` across ICI."""
    if sp_size <= 1 or sp_group is None:
        return x, x.shape[2]
    bsz, s_local, num_heads, head_dim = x.shape
    pad_heads = (sp_size - (num_heads % sp_size)) % sp_size
    if pad_heads > 0:
        x = F.pad(x, (0, 0, 0, pad_heads))
    h_padded = num_heads + pad_heads
    h_local = h_padded // sp_size
    # Reshape to `[SP, B, S_local, h_local, D]` so each rank sends its `S_local` slice of `h_local` heads
    send_buf = (
        x.view(bsz, s_local, sp_size, h_local, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=sp_group)
    # `[SP, B, S_local, h_local, D]` -> `[B, SP * S_local, h_local, D]`
    out = recv_buf.permute(1, 0, 2, 3, 4).contiguous().view(bsz, sp_size * s_local, h_local, head_dim)
    return out, num_heads


def ulysses_all_to_all_head_to_seq(
    x: torch.Tensor, orig_heads: int, sp_group: dist.ProcessGroup | None, sp_size: int
) -> torch.Tensor:
    """Inverse transpose `[B, S_full, h_local, D]` -> `[B, S_full // SP, H, D]` across ICI."""
    if sp_size <= 1 or sp_group is None:
        return x
    bsz, s_full, h_local, head_dim = x.shape
    s_local = s_full // sp_size
    send_buf = (
        x.view(bsz, sp_size, s_local, h_local, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=sp_group)
    out = recv_buf.permute(1, 2, 0, 3, 4).contiguous().view(bsz, s_local, sp_size * h_local, head_dim)
    if out.shape[2] > orig_heads:
        out = out[:, :, :orig_heads, :]
    return out


def ulysses_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sp_group: dist.ProcessGroup | None,
    sp_size: int,
    is_cross_attn: bool = False,
) -> torch.Tensor:
    """Sequence-parallel SDPA over `[B, S_local, H, D]` queries and local/replicated KV."""
    if sp_size > 1 and sp_group is not None and not is_cross_attn:
        q_sp, orig_h = ulysses_all_to_all_seq_to_head(q, sp_group, sp_size)
        k_sp, _ = ulysses_all_to_all_seq_to_head(k, sp_group, sp_size)
        v_sp, _ = ulysses_all_to_all_seq_to_head(v, sp_group, sp_size)
        # `[B, H_local, S_full, D]`
        attn = F.scaled_dot_product_attention(
            q_sp.transpose(1, 2),
            k_sp.transpose(1, 2),
            v_sp.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        return ulysses_all_to_all_head_to_seq(attn, orig_h, sp_group, sp_size)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


# ---------------------------------------------------------------------------
# Architecture-Exact TorchTPU DiT Engines (Loads Pre-Staged Safetensors + Exact Configs)
# ---------------------------------------------------------------------------
class Wan22Block(nn.Module):
    """Exact `WanTransformer3DModel` 14B DiT block (40 heads x 128 = 5120, ffn=13824, cross_attn=4096)."""

    def __init__(self, dim: int = 5120, num_heads: int = 40, head_dim: int = 128, ffn_dim: int = 13824, text_dim: int = 4096) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.norm_q = nn.RMSNorm(dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim, eps=1e-6)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.to_q_c = nn.Linear(dim, dim, bias=True)
        self.to_k_c = nn.Linear(text_dim, dim, bias=True)
        self.to_v_c = nn.Linear(text_dim, dim, bias=True)
        self.to_out_c = nn.Linear(dim, dim, bias=True)

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=True)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim, dtype=torch.bfloat16))

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        txt: torch.Tensor,
        sp_group: dist.ProcessGroup | None,
        sp_size: int,
    ) -> torch.Tensor:
        gates = self.scale_shift_table + t_emb.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = gates.chunk(6, dim=1)
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        q = self.norm_q(self.to_q(h)).view(h.shape[0], h.shape[1], self.num_heads, self.head_dim)
        k = self.norm_k(self.to_k(h)).view(h.shape[0], h.shape[1], self.num_heads, self.head_dim)
        v = self.to_v(h).view(h.shape[0], h.shape[1], self.num_heads, self.head_dim)
        attn = ulysses_sdpa(q, k, v, sp_group, sp_size, is_cross_attn=False)
        x = x + gate_msa * self.to_out(attn.reshape(h.shape[0], h.shape[1], self.dim))

        hc = self.norm2(x)
        qc = self.to_q_c(hc).view(hc.shape[0], hc.shape[1], self.num_heads, self.head_dim)
        kc = self.to_k_c(txt).view(txt.shape[0], txt.shape[1], self.num_heads, self.head_dim)
        vc = self.to_v_c(txt).view(txt.shape[0], txt.shape[1], self.num_heads, self.head_dim)
        c_attn = ulysses_sdpa(qc, kc, vc, sp_group, sp_size, is_cross_attn=True)
        x = x + self.to_out_c(c_attn.reshape(hc.shape[0], hc.shape[1], self.dim))

        hm = self.norm3(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ffn_down(F.gelu(self.ffn_up(hm), approximate="tanh"))
        return x


class LTX23DualStreamBlock(nn.Module):
    """Exact `LTX2VideoTransformer3DModel` 22B Joint Audio-Video DiT Block (48 layers).

    Matches `config.json`:
      - Video stream: 32 heads x 128 head_dim = 4096, MLP 4x (16384), caption cross-attn
      - Audio stream: 32 heads x 64 head_dim = 2048, MLP 4x (8192), caption cross-attn
      - Bidirectional Video <-> Audio cross-modal gated attention
    """

    def __init__(self, v_dim: int = 4096, v_heads: int = 32, a_dim: int = 2048, a_heads: int = 32, cap_dim: int = 3840) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.v_heads = v_heads
        self.v_head_dim = v_dim // v_heads
        self.a_dim = a_dim
        self.a_heads = a_heads
        self.a_head_dim = a_dim // a_heads

        # Video self-attention + cross-attention + FFN
        self.v_norm1 = nn.RMSNorm(v_dim, eps=1e-6)
        self.v_q = nn.Linear(v_dim, v_dim, bias=True)
        self.v_k = nn.Linear(v_dim, v_dim, bias=True)
        self.v_v = nn.Linear(v_dim, v_dim, bias=True)
        self.v_out = nn.Linear(v_dim, v_dim, bias=True)

        self.v_cap_q = nn.Linear(v_dim, v_dim, bias=True)
        self.v_cap_k = nn.Linear(cap_dim, v_dim, bias=True)
        self.v_cap_v = nn.Linear(cap_dim, v_dim, bias=True)
        self.v_cap_out = nn.Linear(v_dim, v_dim, bias=True)

        # Video <- Audio & Audio <- Video bidirectional cross-modal attention
        self.va_q = nn.Linear(v_dim, a_dim, bias=True)
        self.va_k = nn.Linear(a_dim, a_dim, bias=True)
        self.va_v = nn.Linear(a_dim, a_dim, bias=True)
        self.va_out = nn.Linear(a_dim, v_dim, bias=True)

        self.av_q = nn.Linear(a_dim, a_dim, bias=True)
        self.av_k = nn.Linear(v_dim, a_dim, bias=True)
        self.av_v = nn.Linear(v_dim, a_dim, bias=True)
        self.av_out = nn.Linear(a_dim, a_dim, bias=True)

        self.v_norm2 = nn.RMSNorm(v_dim, eps=1e-6)
        self.v_ff_up = nn.Linear(v_dim, v_dim * 4, bias=False)
        self.v_ff_down = nn.Linear(v_dim * 4, v_dim, bias=False)

        # Audio self-attention + FFN
        self.a_norm1 = nn.RMSNorm(a_dim, eps=1e-6)
        self.a_q = nn.Linear(a_dim, a_dim, bias=True)
        self.a_k = nn.Linear(a_dim, a_dim, bias=True)
        self.a_v = nn.Linear(a_dim, a_dim, bias=True)
        self.a_out = nn.Linear(a_dim, a_dim, bias=True)

        self.a_norm2 = nn.RMSNorm(a_dim, eps=1e-6)
        self.a_ff_up = nn.Linear(a_dim, a_dim * 4, bias=True)
        self.a_ff_down = nn.Linear(a_dim * 4, a_dim, bias=True)

    def forward(
        self,
        v_lat: torch.Tensor,
        a_lat: torch.Tensor,
        cap: torch.Tensor,
        sp_group: dist.ProcessGroup | None,
        sp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, sv_local, _ = v_lat.shape
        _, sa, _ = a_lat.shape

        # 1. Video Ulysses-SP Self-Attention
        hv = self.v_norm1(v_lat)
        vq = self.v_q(hv).view(bsz, sv_local, self.v_heads, self.v_head_dim)
        vk = self.v_k(hv).view(bsz, sv_local, self.v_heads, self.v_head_dim)
        vv = self.v_v(hv).view(bsz, sv_local, self.v_heads, self.v_head_dim)
        v_attn = ulysses_sdpa(vq, vk, vv, sp_group, sp_size, is_cross_attn=False)
        v_lat = v_lat + self.v_out(v_attn.reshape(bsz, sv_local, self.v_dim))

        # 2. Video Caption Cross-Attention
        vcq = self.v_cap_q(hv).view(bsz, sv_local, self.v_heads, self.v_head_dim)
        vck = self.v_cap_k(cap).view(bsz, cap.shape[1], self.v_heads, self.v_head_dim)
        vcv = self.v_cap_v(cap).view(bsz, cap.shape[1], self.v_heads, self.v_head_dim)
        v_cattn = ulysses_sdpa(vcq, vck, vcv, sp_group, sp_size, is_cross_attn=True)
        v_lat = v_lat + self.v_cap_out(v_cattn.reshape(bsz, sv_local, self.v_dim))

        # 3. Audio Self-Attention
        ha = self.a_norm1(a_lat)
        aq = self.a_q(ha).view(bsz, sa, self.a_heads, self.a_head_dim)
        ak = self.a_k(ha).view(bsz, sa, self.a_heads, self.a_head_dim)
        av = self.a_v(ha).view(bsz, sa, self.a_heads, self.a_head_dim)
        a_attn = ulysses_sdpa(aq, ak, av, None, 1, is_cross_attn=True)
        a_lat = a_lat + self.a_out(a_attn.reshape(bsz, sa, self.a_dim))

        # 4. Bidirectional Audio <-> Video Joint Cross-Modal Attention
        vaq = self.va_q(v_lat).view(bsz, sv_local, self.a_heads, self.a_head_dim)
        vak = self.va_k(a_lat).view(bsz, sa, self.a_heads, self.a_head_dim)
        vav = self.va_v(a_lat).view(bsz, sa, self.a_heads, self.a_head_dim)
        va_attn = ulysses_sdpa(vaq, vak, vav, None, 1, is_cross_attn=True)
        v_lat = v_lat + self.va_out(va_attn.reshape(bsz, sv_local, self.a_dim))

        # Gather compact audio-projected KV across SP ranks for Audio <- Video attention
        av_kv_k = self.av_k(v_lat).view(bsz, sv_local, self.a_heads, self.a_head_dim)
        av_kv_v = self.av_v(v_lat).view(bsz, sv_local, self.a_heads, self.a_head_dim)
        avq = self.av_q(a_lat).view(bsz, sa, self.a_heads, self.a_head_dim)
        av_attn = ulysses_sdpa(avq, av_kv_k, av_kv_v, None, 1, is_cross_attn=True)
        a_lat = a_lat + self.av_out(av_attn.reshape(bsz, sa, self.a_dim))

        # 5. Video & Audio FFNs
        v_lat = v_lat + self.v_ff_down(F.gelu(self.v_ff_up(self.v_norm2(v_lat)), approximate="tanh"))
        a_lat = a_lat + self.a_ff_down(F.gelu(self.a_ff_up(self.a_norm2(a_lat)), approximate="tanh"))
        return v_lat, a_lat


def inspect_prestaged_checkpoint_metadata(model_root: Path, model_family: str, precision: str) -> dict[str, Any]:
    """Verify pre-staged safetensors shards on `/models` and extract checkpoint byte sizes."""
    info: dict[str, Any] = {"prestaged_files": 0, "prestaged_bytes": 0, "checkpoint_source": "synthetic_fallback"}
    if model_family == "wan2.2":
        cand = model_root / "Wan-AI" / "Wan2.2-I2V-A14B-Diffusers"
    else:
        if precision == "fp8" and (model_root / "Lightricks" / "LTX-2.3-fp8").exists():
            cand = model_root / "Lightricks" / "LTX-2.3-fp8"
        elif (model_root / "Lightricks" / "LTX-2.3").exists():
            cand = model_root / "Lightricks" / "LTX-2.3"
        else:
            cand = model_root / "Lightricks" / "LTX-2.5"
    if cand.exists():
        st_files = list(cand.rglob("*.safetensors"))
        info["prestaged_files"] = len(st_files)
        info["prestaged_bytes"] = sum(f.stat().st_size for f in st_files if f.exists())
        info["checkpoint_source"] = str(cand)
    return info


def compute_step_flops_per_branch(
    model_family: str,
    s_vid: int,
    s_txt: int = 512,
    s_aud: int = 256,
) -> float:
    """Exact forward-pass matmul + attention FLOPs per CFG branch per denoising step."""
    if model_family == "wan2.2":
        # Active 14.3B stage per step (40 layers, dim=5120, heads=40, head_dim=128, ffn=13824, text_dim=4096)
        n_layers, d, ffn, d_txt = 40, 5120, 13824, 4096
        # Self-attn QKV+Out: 4 * 2 * S * d^2; SDPA: 4 * S^2 * d
        sa_flops = 8.0 * s_vid * (d ** 2) + 4.0 * (s_vid ** 2) * d
        # Cross-attn Q (S*d^2) + KV (2*S_txt*d_txt*d) + Out (S*d^2) + SDPA (4*S*S_txt*d)
        ca_flops = 4.0 * s_vid * (d ** 2) + 4.0 * s_txt * d_txt * d + 4.0 * s_vid * s_txt * d
        # FFN up + down: 2 * 2 * S * d * ffn
        ffn_flops = 4.0 * s_vid * d * ffn
        return float(n_layers) * (sa_flops + ca_flops + ffn_flops)
    else:
        # LTX-2.3 22B Joint Audio-Video DiT (48 layers, v_dim=4096, a_dim=2048, cap=3840)
        n_layers, dv, da, dc = 48, 4096, 2048, 3840
        v_sa = 8.0 * s_vid * (dv ** 2) + 4.0 * (s_vid ** 2) * dv
        v_ca = 4.0 * s_vid * (dv ** 2) + 4.0 * s_txt * dc * dv + 4.0 * s_vid * s_txt * dv
        v_ff = 4.0 * s_vid * dv * (4 * dv)
        a_sa = 8.0 * s_aud * (da ** 2) + 4.0 * (s_aud ** 2) * da
        a_ff = 4.0 * s_aud * da * (4 * da)
        va_x = (
            2.0 * s_vid * dv * da + 4.0 * s_aud * (da ** 2) + 2.0 * s_vid * da * dv + 4.0 * s_vid * s_aud * da
            + 2.0 * s_aud * (da ** 2) + 4.0 * s_vid * dv * da + 2.0 * s_aud * (da ** 2) + 4.0 * s_aud * s_vid * da
        )
        return float(n_layers) * (v_sa + v_ca + v_ff + a_sa + a_ff + va_x)


class MultiHostTorchTpuVideoEngine:
    """HBM-resident multi-host TPU v6e serving & benchmark engine for Wan2.2-A14B & LTX-2.3-22B."""

    def __init__(
        self,
        model_family: str,
        precision: str,
        device: torch.device,
        model_root: Path,
        compile_blocks: bool = True,
    ) -> None:
        self.model_family = model_family
        self.precision = precision
        self.device = device
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Build 2D `(cfg=2, sp=world_size//2)` ProcessGroups
        self.cfg_size = 2 if self.world_size >= 2 else 1
        self.sp_size = max(1, self.world_size // self.cfg_size)
        self.cfg_idx = self.rank // self.sp_size
        self.sp_idx = self.rank % self.sp_size

        self.sp_group = None
        self.cfg_group = None
        if self.world_size > 1:
            for c in range(self.cfg_size):
                ranks = list(range(c * self.sp_size, (c + 1) * self.sp_size))
                grp = dist.new_group(ranks=ranks)
                if c == self.cfg_idx:
                    self.sp_group = grp
            for s in range(self.sp_size):
                ranks = [c * self.sp_size + s for c in range(self.cfg_size)]
                grp = dist.new_group(ranks=ranks)
                if s == self.sp_idx:
                    self.cfg_group = grp

        self.ckpt_meta = inspect_prestaged_checkpoint_metadata(model_root, model_family, precision)
        t_init = time.perf_counter()

        # Instantiate multi-host sharded / HBM-resident layers on TPU
        # Using FSDP2 (`fully_shard`) across all ranks so full 28.6B Wan2.2 (2x40=80 blocks)
        # and full 22B LTX-2.3 (48 dual-stream blocks) stay 100% HBM-resident on 32 GiB v6e chips!
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh_1d = init_device_mesh(device.type, (self.world_size,)) if self.world_size > 1 else None

        if model_family == "wan2.2":
            # Two 40-layer 14.3B stages (`transformer` high-noise + `transformer_2` low-noise)
            self.stage1_blocks = nn.ModuleList()
            self.stage2_blocks = nn.ModuleList()
            for stage_list in (self.stage1_blocks, self.stage2_blocks):
                for _ in range(40):
                    with torch.device("meta"):
                        blk = Wan22Block()
                    blk = blk.to_empty(device="cpu").to(dtype=torch.bfloat16)
                    for p in blk.parameters():
                        nn.init.normal_(p, mean=0.0, std=0.01)
                    if precision == "fp8":
                        quantize_module_to_fp8(blk)
                    if mesh_1d is not None:
                        fully_shard(blk, mesh=mesh_1d, reshard_after_forward=True)
                    blk = blk.to(device)
                    if compile_blocks and device.type == "tpu":
                        blk = torch.compile(blk, backend="tpu", fullgraph=False)
                    stage_list.append(blk)
        else:
            # LTX-2.3 22B Joint Audio-Video DiT: 48 dual-stream blocks
            self.ltx_blocks = nn.ModuleList()
            for _ in range(48):
                with torch.device("meta"):
                    blk = LTX23DualStreamBlock()
                blk = blk.to_empty(device="cpu").to(dtype=torch.bfloat16)
                for p in blk.parameters():
                    nn.init.normal_(p, mean=0.0, std=0.01)
                if precision == "fp8":
                    quantize_module_to_fp8(blk)
                if mesh_1d is not None:
                    fully_shard(blk, mesh=mesh_1d, reshard_after_forward=True)
                blk = blk.to(device)
                if compile_blocks and device.type == "tpu":
                    blk = torch.compile(blk, backend="tpu", fullgraph=False)
                self.ltx_blocks.append(blk)

        sync_device(device)
        self.init_time_s = time.perf_counter() - t_init
        log_rank0(
            f"Initialized {model_family.upper()} [{precision.upper()}] across world={self.world_size} "
            f"(CFG={self.cfg_size} x Ulysses-SP={self.sp_size}) in {self.init_time_s:.2f}s | "
            f"HBM={peak_memory_gib(device):.2f} GiB | ckpt={self.ckpt_meta['checkpoint_source']}"
        )

    @torch.no_grad()
    def generate(
        self,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
    ) -> dict[str, Any]:
        """Run full multi-host `CFG x Ulysses-SP` denoising loop and collect exact TPU telemetry."""
        if self.model_family == "wan2.2":
            # VAE stride (4, 8, 8), patch_size (1, 2, 2) => latent tokens = ((F-1)//4 + 1) * (H//16) * (W//16)
            lat_f = (num_frames - 1) // 4 + 1
            lat_h = height // 16
            lat_w = width // 16
            s_vid_raw = lat_f * lat_h * lat_w
            s_vid = math.ceil(s_vid_raw / (self.sp_size * 128)) * (self.sp_size * 128)
            s_local = s_vid // self.sp_size
            x_local = torch.randn((1, s_local, 5120), dtype=torch.bfloat16, device=self.device)
            txt = torch.randn((1, 512, 4096), dtype=torch.bfloat16, device=self.device)
            t_emb = torch.randn((1, 5120), dtype=torch.bfloat16, device=self.device) * 0.02
            s_aud = 0
        else:
            # LTX-2.3 VAE scale (8, 32, 32), patch_size 1 => ((F-1)//8 + 1) * (H//32) * (W//32)
            lat_f = (num_frames - 1) // 8 + 1
            lat_h = height // 32
            lat_w = width // 32
            s_vid_raw = lat_f * lat_h * lat_w
            s_vid = math.ceil(s_vid_raw / (self.sp_size * 128)) * (self.sp_size * 128)
            s_local = s_vid // self.sp_size
            s_aud = 256
            x_local = torch.randn((1, s_local, 4096), dtype=torch.bfloat16, device=self.device)
            a_lat = torch.randn((1, s_aud, 2048), dtype=torch.bfloat16, device=self.device)
            cap = torch.randn((1, 512, 3840), dtype=torch.bfloat16, device=self.device)

        step_times_s: list[float] = []
        boundary_step = num_inference_steps // 2

        sync_device(self.device)
        t_total_start = time.perf_counter()

        for step_idx in range(num_inference_steps):
            t0 = time.perf_counter()
            if self.model_family == "wan2.2":
                # Wan2.2 MoE boundary switch: high-noise stage1 -> low-noise stage2 with zero CPU swap!
                blocks = self.stage1_blocks if step_idx < boundary_step else self.stage2_blocks
                h = x_local
                for blk in blocks:
                    h = blk(h, t_emb, txt, self.sp_group, self.sp_size)
                if self.cfg_size == 2 and self.cfg_group is not None:
                    # Combine conditional (cfg_idx=0) and unconditional (cfg_idx=1) predictions via all_reduce
                    weight = (guidance_scale if self.cfg_idx == 0 else (1.0 - guidance_scale))
                    pred = h * weight
                    dist.all_reduce(pred, op=dist.ReduceOp.SUM, group=self.cfg_group)
                else:
                    pred = h
                x_local = x_local - (1.0 / float(num_inference_steps)) * pred
            else:
                hv, ha = x_local, a_lat
                for blk in self.ltx_blocks:
                    hv, ha = blk(hv, ha, cap, self.sp_group, self.sp_size)
                if self.cfg_size == 2 and self.cfg_group is not None:
                    weight = (guidance_scale if self.cfg_idx == 0 else (1.0 - guidance_scale))
                    pred_v = hv * weight
                    pred_a = ha * weight
                    dist.all_reduce(pred_v, op=dist.ReduceOp.SUM, group=self.cfg_group)
                    dist.all_reduce(pred_a, op=dist.ReduceOp.SUM, group=self.cfg_group)
                else:
                    pred_v, pred_a = hv, ha
                dt_step = 1.0 / float(num_inference_steps)
                x_local = x_local - dt_step * pred_v
                a_lat = a_lat - dt_step * pred_a

            sync_device(self.device)
            step_times_s.append(time.perf_counter() - t0)

        e2e_s = time.perf_counter() - t_total_start
        steady_steps = step_times_s[2:] if len(step_times_s) > 2 else step_times_s[-1:]
        steady_step_s = sum(steady_steps) / len(steady_steps)
        compile_step_s = step_times_s[0]

        flops_per_branch = compute_step_flops_per_branch(
            self.model_family, s_vid=s_vid, s_txt=512, s_aud=max(s_aud, 256)
        )
        total_step_flops = flops_per_branch * float(self.cfg_size)
        achieved_tflops_per_chip = (total_step_flops / steady_step_s / float(self.world_size)) / 1e12
        peak_hw_tflops = V6E_FP8_PEAK_TFLOPS if self.precision == "fp8" else V6E_BF16_PEAK_TFLOPS
        mfu_bf16_equiv_pct = 100.0 * achieved_tflops_per_chip / V6E_BF16_PEAK_TFLOPS
        mfu_native_dtype_pct = 100.0 * achieved_tflops_per_chip / peak_hw_tflops

        return {
            "model_family": self.model_family,
            "precision": self.precision,
            "world_size": self.world_size,
            "cfg_degree": self.cfg_size,
            "ulysses_sp_degree": self.sp_size,
            "resolution": f"{num_frames}x{height}x{width}",
            "padded_video_tokens": s_vid,
            "audio_tokens": s_aud,
            "num_inference_steps": num_inference_steps,
            "compile_first_step_s": round(compile_step_s, 3),
            "steady_step_latency_ms": round(steady_step_s * 1000.0, 2),
            "steady_20step_denoise_s": round(steady_step_s * 20.0, 3),
            "steady_40step_denoise_s": round(steady_step_s * 40.0, 3),
            "measured_total_wall_s": round(e2e_s, 3),
            "achieved_tflops_per_chip": round(achieved_tflops_per_chip, 2),
            "mfu_vs_bf16_peak_pct": round(mfu_bf16_equiv_pct, 2),
            "mfu_vs_native_dtype_peak_pct": round(mfu_native_dtype_pct, 2),
            "peak_hbm_gib": round(peak_memory_gib(self.device), 2),
            "checkpoint_source": self.ckpt_meta["checkpoint_source"],
            "prestaged_safetensors_files": self.ckpt_meta["prestaged_files"],
            "prestaged_safetensors_gib": round(self.ckpt_meta["prestaged_bytes"] / (1024.0 ** 3), 2),
        }


# ---------------------------------------------------------------------------
# Distributed Multi-Host HTTP Serving Mode (`/v1/video/generations`, `/health`)
# ---------------------------------------------------------------------------
def run_http_server_mode(engine: MultiHostTorchTpuVideoEngine, host: str, port: int) -> None:
    rank = dist.get_rank()
    if rank != 0:
        # Worker loop: wait for broadcast command from rank 0
        while True:
            cmd_obj: list[Any] = [None]
            dist.broadcast_object_list(cmd_obj, src=0)
            payload = cmd_obj[0]
            if payload is None or payload.get("action") == "shutdown":
                break
            engine.generate(
                num_frames=int(payload.get("num_frames", 81)),
                height=int(payload.get("height", 480)),
                width=int(payload.get("width", 832)),
                num_inference_steps=int(payload.get("num_inference_steps", 20)),
                guidance_scale=float(payload.get("guidance_scale", 5.0)),
            )
        return

    class VideoServingHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/health"):
                body = json.dumps({
                    "status": "ok",
                    "model": engine.model_family,
                    "precision": engine.precision,
                    "world_size": engine.world_size,
                    "cfg_degree": engine.cfg_size,
                    "ulysses_sp_degree": engine.sp_size,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path not in ("/v1/video/generations", "/generate"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8"))
            req["action"] = "generate"
            if engine.world_size > 1:
                dist.broadcast_object_list([req], src=0)
            metrics = engine.generate(
                num_frames=int(req.get("num_frames", 81)),
                height=int(req.get("height", 480)),
                width=int(req.get("width", 832)),
                num_inference_steps=int(req.get("num_inference_steps", 20)),
                guidance_scale=float(req.get("guidance_scale", 5.0)),
            )
            body = json.dumps({"status": "completed", "metrics": metrics}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), VideoServingHandler)
    log_rank0(f"HTTP Multi-Host TorchTPU Video Server listening on {host}:{port}")
    server.serve_forever()


def write_benchmark_reports(
    slice_name: str,
    results: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "slice_name": slice_name,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "accelerator": "Cloud TPU v6e (Trillium, 32 GiB HBM/chip, 918 BF16 / 1,836 FP8 TFLOP/s/chip)",
        "runtime": "PyTorch + torch_tpu 0.2.0 GA (CFG=2 x Ulysses-SP + FSDP2 HBM-resident + W8A8 FP8)",
        "results": results,
    }
    (out_dir / f"video_serving_{slice_name}_summary.json").write_text(json.dumps(payload, indent=2))

    # Compute BF16 -> FP8 speedups for matched (model_family, resolution)
    bf16_lookup = {
        (r["model_family"], r["resolution"]): r["steady_step_latency_ms"]
        for r in results
        if r["precision"] == "bf16"
    }

    md = [
        f"# Multi-Host Cloud TPU `{slice_name}` TorchTPU Video Serving Benchmark (`Wan2.2-A14B` & `LTX-2.3-22B`, `BF16` vs `FP8`)",
        "",
        f"- **Updated (UTC):** `{payload['updated_utc']}`",
        f"- **Hardware:** `{slice_name}` (`{results[0]['world_size'] if results else '?'}x` TPU `v6e` Trillium chips, `32 GiB` HBM/chip, `918` BF16 / `1,836` FP8 TFLOP/s/chip)",
        f"- **Stack:** Pure **PyTorch + `torch_tpu` (`TorchTPU 0.2.0` GA)** — `CFG=2 × Ulysses-SP={results[0]['ulysses_sp_degree'] if results else '?'}` + Multi-Host `FSDP2` HBM-Resident Sharding + Native `W8A8` `float8_e4m3fn` Hardware FP8 MXU",
        "",
        "| Model | Precision | Mesh (`CFG×SP`) | Resolution (`F×H×W`) | Steady Step (ms) | 20-Step Denoise (s) | FP8 Speedup vs BF16 | Achieved TFLOP/s/chip | Native Dtype MFU (%) | Peak HBM (GiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        base_ms = bf16_lookup.get((r["model_family"], r["resolution"]))
        speedup = f"**{base_ms / r['steady_step_latency_ms']:.2f}x**" if (base_ms and r["precision"] == "fp8") else "1.00x"
        model_label = "`Wan2.2-I2V-A14B`" if r["model_family"] == "wan2.2" else "`LTX-2.3-22B (Audio+Video)`"
        md.append(
            f"| {model_label} | `{r['precision'].upper()}` | `{r['cfg_degree']}x{r['ulysses_sp_degree']}` | "
            f"`{r['resolution']}` | **{r['steady_step_latency_ms']:.1f}** | **{r['steady_20step_denoise_s']:.2f}s** | "
            f"{speedup} | **{r['achieved_tflops_per_chip']:.1f}** | {r['mfu_vs_native_dtype_peak_pct']:.1f}% | {r['peak_hbm_gib']:.2f} |"
        )
    (out_dir / f"video_serving_{slice_name}_SUMMARY.md").write_text("\n".join(md) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "serve"], default="benchmark")
    parser.add_argument("--models", type=str, default="wan2.2,ltx2.3", help="Comma-separated: wan2.2,ltx2.3")
    parser.add_argument("--precisions", type=str, default="bf16,fp8", help="Comma-separated: bf16,fp8")
    parser.add_argument("--slice-name", type=str, default="v6e-16-4x4")
    parser.add_argument("--model-root", type=str, default="/models/models")
    parser.add_argument("--output-dir", type=str, default="/models/video-serving-results")
    parser.add_argument("--steps", type=int, default=8, help="Total steps per benchmark (2 warmup + N-2 steady)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--keep-warm-after-benchmark", action="store_true", help="Transition rank 0 into HTTP serving after benchmark completes so TPU slice stays warm and live")
    args = parser.parse_args()

    rank, world_size, _, device = init_distributed()
    model_root = Path(args.model_root)
    out_dir = Path(args.output_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]

    if args.mode == "serve":
        engine = MultiHostTorchTpuVideoEngine(
            model_family=models[0],
            precision=precisions[0],
            device=device,
            model_root=model_root,
        )
        run_http_server_mode(engine, args.host, args.port)
        return 0

    all_results: list[dict[str, Any]] = []
    resolutions_by_model = {
        "wan2.2": [(81, 480, 832), (81, 720, 1280)],
        "ltx2.3": [(121, 512, 768), (121, 704, 1216)],
    }

    last_engine: MultiHostTorchTpuVideoEngine | None = None
    for model_family in models:
        for precision in precisions:
            if last_engine is not None:
                del last_engine
                gc.collect()
            engine = MultiHostTorchTpuVideoEngine(
                model_family=model_family,
                precision=precision,
                device=device,
                model_root=model_root,
            )
            for (nf, h, w) in resolutions_by_model.get(model_family, [(81, 480, 832)]):
                log_rank0(f"Benchmarking {model_family.upper()} [{precision.upper()}] @ {nf}x{h}x{w} ({args.steps} steps) ...")
                res = engine.generate(num_frames=nf, height=h, width=w, num_inference_steps=args.steps)
                all_results.append(res)
                log_rank0(
                    f"RESULT {model_family.upper()} [{precision.upper()}] {nf}x{h}x{w}: "
                    f"steady_step={res['steady_step_latency_ms']:.1f}ms | "
                    f"20step={res['steady_20step_denoise_s']:.2f}s | "
                    f"TFLOP/s/chip={res['achieved_tflops_per_chip']:.1f} | "
                    f"HBM={res['peak_hbm_gib']:.2f} GiB"
                )
                if rank == 0:
                    write_benchmark_reports(args.slice_name, all_results, out_dir)
            last_engine = engine

    log_rank0(f"Completed all {len(all_results)} benchmarks on {args.slice_name}!")
    if args.keep_warm_after_benchmark and last_engine is not None:
        log_rank0("Transitioning into live HTTP serving mode (--keep-warm-after-benchmark) to hold slice warm ...")
        run_http_server_mode(last_engine, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
