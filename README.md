# Multi-Host Video Diffusion Serving Recipe on Google Cloud TPU v6e

Multi-host PyTorch and `torch_tpu` (`TorchTPU 0.2.0`) serving harness, 2D `(CFG × Ulysses-SP)` + `FSDP2` sharding engine, and GKE provisioning manifests for `Wan2.2-I2V-A14B` and `LTX-2.3` on Google Cloud TPU `v6e-16` and `v6e-32`.

**Summary:** This repository provides the complete multi-host TorchTPU serving and benchmarking recipe to fit `Wan2.2-I2V-A14B` (`28.6B` total DiT parameters) and `LTX-2.3` (`22B` joint audio-video DiT) inside TPU v6e's `32 GiB`/chip HBM without CPU offload (`~1.8–3.6 GiB/chip` sharded weights in `bf16` and `~0.9–1.8 GiB/chip` in `W8A8 fp8`). **Live `v6e-16` and `v6e-32` slice latency runs have not yet been captured in [`results/`](./results/)**, so only verified sharding configurations, token geometries, and per-chip weight footprints are reported below (zero unmeasured latency projections).

---

## 1. Model, Framework, and Hardware

- **Models & Weight Precisions:**
  - **Wan2.2-I2V-A14B (`Wan-AI/Wan2.2-I2V-A14B-Diffusers`):** Two-stage Mixture-of-Experts video DiT (`14.3B` high-noise `transformer` + `14.3B` low-noise `transformer_2` = `28.6B` total DiT parameters across `2 × 40` blocks, width `5,120`, `40` heads × `128` dim, FFN `13,824`), supported in `bfloat16` and dynamic `W8A8` `float8_e4m3fn` (`TpuW8A8FP8Linear`).
  - **LTX-2.3 (`Lightricks/LTX-2.3` & `Lightricks/LTX-2.3-fp8`):** `22B`-parameter asymmetric dual-stream audio-video DiT (`48` blocks; video stream `32 × 128 = 4,096`, stereo-audio stream `32 × 64 = 2,048`, plus bidirectional cross-modal attention), supported in `bfloat16` (`ltx-2.3-22b-dev.safetensors`) and `float8_e4m3fn` (`ltx-2.3-22b-dev-fp8.safetensors`).
- **Framework:** Python 3.14, PyTorch 2.15.0, `torch_tpu` `0.2.0` (`c10d` / `tpu` process group, `FSDP2` `fully_shard`, `torch.compile(backend="tpu")`, `torch._scaled_mm` FP8 dispatch; [`docker/Dockerfile`](./docker/Dockerfile)).
- **Hardware Target:** Google Cloud TPU v6e (Trillium) slices on GKE (`32 GiB` HBM per chip, `918` `bf16` / `1,836` `fp8` peak TFLOP/s per chip, `4` chips per host):
  - `v6e-16` (`4x4` ICI topology: `4` hosts × `4` chips = `16` chips, `512 GiB` aggregate HBM)
  - `v6e-32` (`4x8` ICI topology: `8` hosts × `4` chips = `32` chips, `1,024 GiB` aggregate HBM)

---

## 2. Main Results

### Verified Sharding Matrix & Per-Chip Weight Footprint (Pending Live `v6e` Slice Timing)

Because each TPU `v6e` chip has `32 GiB` HBM (vs. `95 GiB` on `v5p` and `192 GiB` on `v7x`), unsharded `bf16` checkpoints (`57.2 GB` for dual-stage `Wan2.2-A14B`, `46.2 GB` for `LTX-2.3`) exceed single-chip memory. [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py) combines a 2D `(CFG=2, Ulysses-SP)` mesh with `FSDP2` `fully_shard` across all slice ranks:

| Model & Precision | Slice Topology | 2D Mesh (`CFG × SP`) | Evaluation Shapes (`F × H × W`) | Padded Video Tokens / Branch | Sharded DiT Weights / Chip | Measured Timing Status |
| :--- | :--- | :---: | :--- | ---: | ---: | :--- |
| `Wan2.2-I2V-A14B` (`bf16`) | `v6e-16` (`4x4`, 16 chips) | `2 × 8` | `81×480×832` / `81×720×1280` | `32,768` / `75,776` | ~3.58 GiB (`50%` less in `fp8`: ~1.79 GiB) | Pending `v6e-16` slice run |
| `Wan2.2-I2V-A14B` (`bf16` / `fp8`) | `v6e-32` (`4x8`, 32 chips) | `2 × 16` | `81×480×832` / `81×720×1280` | `32,768` / `75,776` | ~1.79 GiB (`bf16`) / ~0.90 GiB (`fp8`) | Pending `v6e-32` slice run |
| `LTX-2.3-22B` (`bf16`) | `v6e-16` (`4x4`, 16 chips) | `2 × 8` | `121×512×768` / `121×704×1216` | `6,144` / `13,312` (+`256` audio) | ~2.88 GiB (`50%` less in `fp8`: ~1.44 GiB) | Pending `v6e-16` slice run |
| `LTX-2.3-22B` (`bf16` / `fp8`) | `v6e-32` (`4x8`, 32 chips) | `2 × 16` | `121×512×768` / `121×704×1216` | `6,144` / `14,336` (+`256` audio) | ~1.44 GiB (`bf16`) / ~0.72 GiB (`fp8`) | Pending `v6e-32` slice run |

*Notes:* Sharded weight footprints reflect `FSDP2` parameter partitioning across 16 or 32 ranks (`W8A8` `float8_e4m3fn` stores `1 byte/parameter` vs. `2 bytes/parameter` in `bfloat16`). Note that [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py) benchmarks architecture-exact DiT transformer blocks (`Wan22Block` × 80 and `LTX23DualStreamBlock` × 48) without text-encoder or 3D VAE decode passes; see [`APPENDIX.md`](./APPENDIX.md) for details.

---

## 3. Reproducing the Benchmark & HTTP Serving Deployment

Manifests and scripts: [`k8s/model-prestage-jobs.yaml`](./k8s/model-prestage-jobs.yaml), [`k8s/v6e-warm-infra.yaml`](./k8s/v6e-warm-infra.yaml), [`k8s/v6e-multihost-video-serving-jobs.yaml`](./k8s/v6e-multihost-video-serving-jobs.yaml), and [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py):

```bash
# 1. Stage Wan2.2-I2V-A14B and LTX-2.3 (bf16 + fp8) safetensors to your regional GCS bucket
kubectl apply -f k8s/model-prestage-jobs.yaml

# 2. (Optional) Deploy Spot + cascading Flex-Start warm slice holders and autoscaler node pinners
kubectl apply -f k8s/v6e-warm-infra.yaml

# 3. Create the serving script ConfigMap and launch multi-host v6e-16 (4x4) and v6e-32 (4x8) Indexed Jobs
kubectl create configmap v6e-multihost-video-serving-script \
  --from-file=run_v6e_multihost_video_serving.py=scripts/run_v6e_multihost_video_serving.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/v6e-multihost-video-serving-jobs.yaml
```

Once rank 0 completes the benchmark sweep (`--keep-warm-after-benchmark`), it transitions into an HTTP endpoint on `:8080` (`GET /health`, `POST /v1/video/generations`) and writes JSON summaries to `--output-dir`.

---

## 4. Detailed Architecture Notes and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Detailed design of the 2D `(CFG=2, Ulysses-SP)` communication groups, non-divisible head padding (`40 -> 48` heads at `SP=16`), `TpuW8A8FP8Linear` quantization rules, cascading `7d -> 4d -> 3d -> 2d` GKE `ProvisioningRequest` controller, and DiT-only scope caveats.
- **[`results/`](./results/):** Output directory for `video_serving_<slice>_summary.json` artifacts produced by [`scripts/run_v6e_multihost_video_serving.py`](./scripts/run_v6e_multihost_video_serving.py).
