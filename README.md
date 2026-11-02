# Multi-Host TPU `v6e` Video Diffusion Serving (`TorchTPU 0.2.0`)

**PyTorch + `torch_tpu` (`TorchTPU 0.2.0` GA)** multi-host video generation serving and benchmarking on **Cloud TPU `v6e-16` (`4x4`, 4 hosts, 16 chips, `512 GiB` HBM)** and **Cloud TPU `v6e-32` (`4x8`, 8 hosts, 32 chips, `1,024 GiB` HBM)** for:

1. **[`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)** — Two-stage Mixture-of-Experts (`14.3B` High-Noise `transformer` + `14.3B` Low-Noise `transformer_2` = `28.6B` total DiT parameters, `126.1 GB` staged checkpoint) in **`BF16` (`918` TFLOP/s/chip) and `W8A8 FP8` (`1,836` TFLOP/s/chip)**.
2. **[`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) & [`Lightricks/LTX-2.3-fp8`](https://huggingface.co/Lightricks/LTX-2.3-fp8)** — `22B` Asymmetric Dual-Stream Joint Audio-Video DiT (`48` blocks, `4,096` video width + `2,048` stereo-audio width + bidirectional AV cross-attention) in **`BF16` (`ltx-2.3-22b-dev.safetensors`) and `FP8 e4m3fn` (`ltx-2.3-22b-dev-fp8.safetensors`)**.

---

## 1. Multi-Host TPU `v6e` Architecture & Implementation

| Capability | `v6e-16` (`4x4`, 16 Chips) | `v6e-32` (`4x8`, 32 Chips) | Script / Manifest Anchor | Rationale on `v6e` (`32 GiB` HBM/Chip) |
|---|---|---|---|---|
| **2D Inference Mesh `(cfg_parallel, ulysses_sp)`** | **`CFG=2 × Ulysses-SP=8`** (`2` 8-chip ICI subgroups) | **`CFG=2 × Ulysses-SP=16`** (`2` 16-chip ICI subgroups) | [`run_v6e_multihost_video_serving.py#L20`](./scripts/run_v6e_multihost_video_serving.py#L20-L35) | Avoids `2×` sequential Classifier-Free Guidance latency while splitting `S = 32,768..75,600` across 8 or 16 chips via head-sharded `all_to_all_single` Ulysses Attention |
| **Zero-CPU-Offload HBM Residency (`FSDP2`)** | `fully_shard` across all `16` ranks (`~3.6 GiB/chip` BF16 / `~1.8 GiB/chip` FP8 for both `Wan2.2` stages) | `fully_shard` across all `32` ranks (`~1.8 GiB/chip` BF16 / `~0.9 GiB/chip` FP8) | [`run_v6e_multihost_video_serving.py#L22`](./scripts/run_v6e_multihost_video_serving.py#L22-L35) | Keeps both `Wan2.2` High-Noise & Low-Noise `14.3B` stages resident in TPU HBM (`0 ms` boundary handoff, no host PCIe offload) |
| **Native `W8A8` Hardware `FP8` MXU** | `TpuW8A8FP8Linear` (`float8_e4m3fn` × `float8_e4m3fn` -> `BF16` + `weight_scale`) | **`2×` MXU (`1,836` vs `918` TFLOP/s/chip = `58.75` PF/s on `v6e-32`)** | [`TpuW8A8FP8Linear#L132`](./scripts/run_v6e_multihost_video_serving.py#L132-L210) | Consumes `Lightricks/LTX-2.3-fp8` and dynamically quantizes `Wan2.2-I2V-A14B` DiT linear projections while keeping `AdaLN`, norms, and 3D RoPE in high precision |
| **Static XLA Graph Caching (`torch.compile`)** | Bucketed `(F, H, W)` shapes + tensor `timestep` / `boundary` inputs | Zero graph recompilation across `20–40` denoising steps & HTTP `/v1/video/generations` requests | [`torch.compile#L599`](./scripts/run_v6e_multihost_video_serving.py#L599-L615) · [`HTTP#L743`](./scripts/run_v6e_multihost_video_serving.py#L743-L840) | Avoids dynamic Python scalar specialization on diffusion step indices |
| **Warm Multi-Host Slice Pinning** | `PriorityClass` preemption (`tpu-warm-placeholder` -> `tpu-serving-workload`) | Automatic `Spot` + `Flex-Start` `ProvisioningRequest` cascade + `pin-v6e-nodes` DaemonSet | [`k8s/v6e-warm-infra.yaml`](./k8s/v6e-warm-infra.yaml) · [`k8s/v6e-multihost-video-serving-jobs.yaml`](./k8s/v6e-multihost-video-serving-jobs.yaml) | Prevents GKE autoscaler scale-down between runs and keeps `torch_tpu` + compiled XLA caches resident in HBM |

---

## 2. Repository Structure & Quickstart

- [`docker/Dockerfile`](./docker/Dockerfile) — Python 3.14 + PyTorch + `TorchTPU 0.2.0` GA container
- [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py) — Multi-host TorchTPU engine (`Wan2.2-I2V-A14B` & `LTX-2.3`, `BF16` + `W8A8 FP8`, benchmark + HTTP server)
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel Hugging Face -> GCS safetensors stager
- [`k8s/model-prestage-jobs.yaml`](./k8s/model-prestage-jobs.yaml) — K8s Jobs to stage `Wan2.2-I2V-A14B` & `LTX-2.3` (`BF16` + `FP8`) to GCS
- [`k8s/v6e-warm-infra.yaml`](./k8s/v6e-warm-infra.yaml) — `PriorityClasses`, `Spot`+`Flex-Start` cascade controller, warm placeholders & node pinners
- [`k8s/v6e-multihost-video-serving-jobs.yaml`](./k8s/v6e-multihost-video-serving-jobs.yaml) — 4-host (`v6e-16` `4x4`) & 8-host (`v6e-32` `4x8`) Indexed Jobs + HTTP Serving endpoint

```bash
# 1. Stage Wan-AI/Wan2.2-I2V-A14B-Diffusers and Lightricks/LTX-2.3 (BF16 + FP8) into GCS
kubectl apply -f k8s/model-prestage-jobs.yaml

# 2. Deploy Warm Slice Pinning + Flex-Start Cascade Infrastructure
kubectl apply -f k8s/v6e-warm-infra.yaml

# 3. Create Serving Script ConfigMap & Launch Multi-Host v6e-16 (4x4) and v6e-32 (4x8) Jobs
kubectl create configmap v6e-multihost-video-serving-script \
  --from-file=run_v6e_multihost_video_serving.py=scripts/run_v6e_multihost_video_serving.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/v6e-multihost-video-serving-jobs.yaml
```
