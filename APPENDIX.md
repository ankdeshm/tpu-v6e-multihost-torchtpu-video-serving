# Appendix: Multi-Host Sharding, W8A8 FP8 Design, and Provisioning Infrastructure (TPU v6e)

This document describes the multi-host TorchTPU communication layout, `FSDP2` weight residency, `W8A8` `float8_e4m3fn` linear quantization, warm slice provisioning controller, and scope notes for [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py). For the main overview and reproduction steps, see [`README.md`](./README.md).

---

## 1. 2D `(CFG × Ulysses-SP)` Mesh & `FSDP2` HBM Residency on `32 GiB` Chips

1. **Classifier-Free Guidance (`cfg = 2`) + Ulysses Sequence Parallelism (`sp = world_size // 2`):**
   - Ranks `[0 .. SP-1]` compute the conditional diffusion branch (`cfg_idx = 0`) while ranks `[SP .. 2·SP-1]` compute the unconditional branch (`cfg_idx = 1`) concurrently, combining weighted branch predictions at the end of each denoising step with a 2-rank `dist.all_reduce` across `cfg_group`.
   - Within each `SP` subgroup (`sp = 8` on `v6e-16`, `sp = 16` on `v6e-32`), latent video sequences `[B, S_local, H, D]` are transposed into head-sharded tensors `[B, S_full, H_pad // SP, D]` via `dist.all_to_all_single` (`ulysses_all_to_all_seq_to_head` and `ulysses_all_to_all_head_to_seq`), evaluated with `F.scaled_dot_product_attention`, and transposed back.
   - **Automatic Head Padding for `SP = 16`:** `Wan2.2-A14B` uses `40` attention heads (`40 × 128 = 5,120`), which is divisible by `8` (`v6e-16`) but not by `16` (`v6e-32`). `ulysses_all_to_all_seq_to_head` pads the head axis from `40` to `48` (`3` heads/rank on `SP=16`) before `all_to_all_single` and slices back to `:40` after the inverse exchange.
2. **Zero-Swap Dual-Stage `FSDP2` Residency (`fully_shard`):**
   - `Wan2.2-I2V-A14B` alternates between two distinct `14.3B` 40-layer transformers (`stage1_blocks` for high-noise steps `< N // 2` and `stage2_blocks` for low-noise steps `>= N // 2`). Wrapping every block with `torch.distributed.fsdp.fully_shard(..., reshard_after_forward=True)` across the full 16-chip or 32-chip mesh keeps both stages resident in TPU HBM (`~3.58 GiB/chip` on `v6e-16` `bf16`; `~1.79 GiB/chip` on `v6e-32` `bf16`) so the midpoint stage switch incurs zero host-to-device transfer stalls.
   - For `LTX-2.3-22B`, each of the `48` `LTX23DualStreamBlock` modules executes video self-attention (`Ulysses-SP`), video caption cross-attention, audio self-attention (`256` tokens), bidirectional video-audio cross-attention, and dual-stream FFNs under `fully_shard`.
3. **Per-Block `torch.compile(backend="tpu")`:**
   - Video token counts are rounded up to multiples of `SP × 128` (`s_vid = ceil(s_raw / (SP · 128)) · (SP · 128)`) so XLA compiled block graphs remain static across all denoising steps and repeated HTTP `/v1/video/generations` calls.

---

## 2. Native `W8A8` `float8_e4m3fn` Linear Quantization (`TpuW8A8FP8Linear`)

- `quantize_module_to_fp8()` replaces `nn.Linear` layers whose `in_features` and `out_features` are multiples of `16` with `TpuW8A8FP8Linear`, while keeping sensitive modulation and boundary layers (`patch_embed`, `proj_out`, `norm`, `time_embed`, `adaln`, `scale_shift_table`) in `bfloat16` / `float32`.
- Weights are stored in `torch.float8_e4m3fn` (`FP8_MAX = 448.0`) with an `fp32` `weight_scale` buffer; activations are dynamically scaled per tensor and dispatched through `torch._scaled_mm(..., out_dtype=torch.bfloat16)` onto TPU `v6e` Trillium's `1,836 TFLOP/s` FP8 MXUs (with a fallback XLA `float8 -> bfloat16` `matmul` path if `_scaled_mm` raises on a specific shape).

---

## 3. Warm Multi-Host Slice Pinning & Cascading Flex-Start Controller (`k8s/v6e-warm-infra.yaml`)

Multi-host `v6e-16` (`4x4`) and `v6e-32` (`4x8`) slices can face queuing delays or GKE cluster-autoscaler scale-down between benchmark runs. [`k8s/v6e-warm-infra.yaml`](./k8s/v6e-warm-infra.yaml) deploys three coordination primitives:

1. **Preemptible Warm Holders (`PriorityClass` `-100` vs. `1000`):** Low-priority `tpu-warm-holder` `pause` pods hold `4` TPU chips per host on Spot and Flex-Start pools and yield immediately (`PreemptLowerPriority`) when `tpu-serving-workload` Indexed Jobs (`v6e-16-torchtpu-video-bench-spot` or `v6e-32-torchtpu-video-bench-spot`) are scheduled.
2. **Zero-TPU Node Pinner (`pin-v6e-eb-nodes` `DaemonSet`):** Places a `safe-to-evict: "false"` `10m`-CPU pod on every provisioned `v6e` node so GKE's autoscaler does not terminate warm hosts during brief gaps between jobs.
3. **Cascading `ProvisioningRequest` Duration Controller (`flex-cascade-controller`):** Monitors `queued-provisioning.gke.io` `ProvisioningRequest` objects on `tpu-v6e-32-flex-pool` and `tpu-v6e-16-flex-pool`, automatically stepping `maxRunDurationSeconds` down across `7d (604,800 s) -> 4d (345,600 s) -> 3d (259,200 s) -> 2d (172,800 s)` when a longer reservation window encounters `BookingFailed` or exceeds its wait budget.

---

## 4. Scope & Measurement Caveats

1. **No Committed Hardware Logs Yet:** [`results/`](./results/) does not yet contain `video_serving_<slice>_summary.json` outputs from a live `v6e-16` or `v6e-32` run; no estimated latencies or MFU numbers are reported in `README.md`.
2. **DiT-Core Benchmark Scope:** `MultiHostTorchTpuVideoEngine` in [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py) initializes architecture-exact `Wan22Block` (`80` blocks) and `LTX23DualStreamBlock` (`48` blocks) modules on `meta` -> `cpu` -> `FSDP2` `tpu` to measure steady-state multi-host DiT denoising latency, FLOP/s, and HBM usage (inspecting pre-staged safetensors metadata on `/models` when present, and excluding text-encoder prompt encoding, 3D VAE decoding, and MP4 container muxing).
3. **Container Build Requirement (`docker/Dockerfile`):** Note that [`docker/Dockerfile`](./docker/Dockerfile) copies a helper script (`patch_flint_configs.py`) from the TorchTitan build context; in GKE manifests ([`k8s/v6e-multihost-video-serving-jobs.yaml`](./k8s/v6e-multihost-video-serving-jobs.yaml)), the prebuilt `torchtpu` image is referenced via `YOUR_GCP_PROJECT_ID`.
