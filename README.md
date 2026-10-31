# Multi-Host Cloud TPU `v6e` (`4x4` & `4x8`) Video Generation Serving & Benchmarking with PyTorch + `TorchTPU 0.2.0` GA

Pure **PyTorch + `torch_tpu` (`TorchTPU 0.2.0` GA)** (zero JAX) multi-host video generation serving and benchmarking on **Cloud TPU `v6e-16` (`4x4`, 4 hosts, 16 Trillium chips, `512 GiB` HBM)** and **Cloud TPU `v6e-32` (`4x8`, 8 hosts, 32 Trillium chips, `1,024 GiB` HBM)** for:

1. **[`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)** — Two-stage Mixture-of-Experts (`14.3B` High-Noise `transformer` + `14.3B` Low-Noise `transformer_2` = `28.6B` total DiT parameters, `126.1 GB` staged checkpoint) in **both `BF16` (`918` TFLOP/s/chip) and `W8A8 FP8` (`1,836` TFLOP/s/chip)**.
2. **[`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) & [`Lightricks/LTX-2.3-fp8`](https://huggingface.co/Lightricks/LTX-2.3-fp8)** — `22B` Asymmetric Dual-Stream Joint Audio-Video DiT (`48` blocks, `4,096` video width + `2,048` stereo-audio width + bidirectional AV cross-attention) in **both official `BF16` (`ltx-2.3-22b-dev.safetensors`) and official native `FP8 e4m3fn` (`ltx-2.3-22b-dev-fp8.safetensors`)**.

---

## 1. Multi-Host TPU `v6e` (`32 GiB/chip`) Architecture & Parallelism Design

| Capability | `v6e-16` (`4x4`, 16 Chips) | `v6e-32` (`4x8`, 32 Chips) | Why Critical on `v6e` (`32 GiB` HBM/Chip) |
|---|---|---|---|
| **2D Inference Mesh `(cfg_parallel, ulysses_sp)`** | **`CFG=2 × Ulysses-SP=8`** (`2` 8-chip ICI subgroups) | **`CFG=2 × Ulysses-SP=16`** (`2` 16-chip ICI subgroups) | Eliminates `2×` sequential Classifier-Free Guidance latency while splitting sequence `S` (`32,768` at `81×480×832` or `75,600` at `720p`) across 8 or 16 chips via head-sharded `all_to_all_single` Ulysses Attention |
| **Zero-CPU-Offload HBM Residency (`FSDP2`)** | `fully_shard` across all `16` ranks (`~3.6 GiB/chip` BF16 / `~1.8 GiB/chip` FP8 for both `Wan2.2` High+Low stages simultaneously!) | `fully_shard` across all `32` ranks (`~1.8 GiB/chip` BF16 / `~0.9 GiB/chip` FP8) | Keeps **both** `Wan2.2` High-Noise & Low-Noise `14.3B` stages simultaneously resident in TPU HBM (`0 ms` boundary handoff, zero host PCIe offload!) |
| **Native Trillium `W8A8` Hardware `FP8` MXU** | `TpuW8A8FP8Linear` (`torch.float8_e4m3fn` × `torch.float8_e4m3fn` -> `BF16` dot_general + per-tensor `weight_scale`) | **`2×` peak MXU (`1,836` vs `918` TFLOP/s/chip = `58.75` PFLOP/s on `v6e-32`)** + **`2×` smaller ICI `AllGather` bandwidth** | Directly consumes official `Lightricks/LTX-2.3-fp8` (`e4m3fn` + `weight_scale`) and dynamically quantizes `Wan2.2-I2V-A14B` DiT linear projections while keeping `AdaLN`, norms, embeddings, and 3D RoPE in high precision |
| **Static XLA Graph Caching (`torch.compile(backend="openxla")`)** | Bucketed `(F, H, W)` shapes + tensor `timestep` / `boundary` inputs | Zero graph recompilation across all `20–40` denoising steps and across repeated HTTP `/v1/video/generations` requests | Avoids dynamic Python scalar specialization on diffusion step indices |
| **Warm Multi-Host Slice Pinning** | `PriorityClass` preemption (`tpu-warm-placeholder` -> `tpu-serving-workload`) + `--keep-warm-after-benchmark` | Automatic `Spot` + `Flex-Start` `ProvisioningRequest` cascade controller + `pin-v6e-nodes` DaemonSet | Prevents GKE autoscaler scale-down between runs and keeps `torch_tpu` + compiled XLA caches hot in HBM |

---

## 2. Repository Structure & Quickstart

```text
├── README.md
├── docker/
│   └── Dockerfile                              # Python 3.14 + PyTorch + TorchTPU 0.2.0 GA container
├── scripts/
│   ├── run_v6e_multihost_video_serving.py      # Multi-host TorchTPU engine (Wan2.2-I2V-A14B & LTX-2.3, BF16 + W8A8 FP8, benchmark + HTTP server)
│   └── stage_hf_to_gcs.py                      # Parallel Hugging Face -> GCS safetensors stager
├── k8s/
│   ├── model-prestage-jobs.yaml                # K8s Jobs to stage Wan2.2-I2V-A14B & LTX-2.3 (BF16 + FP8) to GCS
│   ├── v6e-warm-infra.yaml                     # PriorityClasses, Spot+Flex-Start cascade controller, warm placeholders & node pinners
│   └── v6e-multihost-video-serving-jobs.yaml   # 4-host (v6e-16 4x4) & 8-host (v6e-32 4x8) Indexed Jobs + HTTP Serving endpoint
└── results/                                    # Auto-synced JSON & Markdown benchmark outputs (BF16 vs FP8 latency, step time, MFU, HBM)
```

```bash
# 1. Stage Wan-AI/Wan2.2-I2V-A14B-Diffusers and Lightricks/LTX-2.3 (BF16 + FP8) into GCS
kubectl apply -f k8s/model-prestage-jobs.yaml

# 2. Deploy Warm Slice Pinning + Flex-Start Cascade Infrastructure
kubectl apply -f k8s/v6e-warm-infra.yaml

# 3. Create Serving Script ConfigMap & Launch Multi-Host v6e-16 (4x4) and v6e-32 (4x8) BF16 + FP8 Benchmarks + Warm HTTP Server
kubectl create configmap v6e-multihost-video-serving-script \
  --from-file=run_v6e_multihost_video_serving.py=scripts/run_v6e_multihost_video_serving.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/v6e-multihost-video-serving-jobs.yaml
```
